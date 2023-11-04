import open3d as o3d
from torch.utils.data import Subset, Dataset, DataLoader
import os
import os.path as osp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
import torch
import time
import copy
import pytorch_lightning as pl
import requests
from loguru import logger
from omegaconf import OmegaConf
from typing import Tuple
from common.datamodels import ActionTypeDef
from torch.utils.data import WeightedRandomSampler
from learning.datasets.weighted_sampler import SampleWeightsGenerator
import learning.datasets.augmentation as aug
import MinkowskiEngine as ME

class RuntimeDatasetReal(Dataset):
    """RuntimeDatasetReal is a real-world dataset
    that is used to filter and generate data samples at runtime."""

    def __init__(self,
                 # logging params
                 logging_dir: str = './log',
                 namespace: str = 'experiment_real',
                 tag: str = 'debug',
                 episode_range: tuple = (0, 20),
                 # data augmentation
                 use_augmentation: bool = True,
                 normal_aug_types: tuple = ('depth', 'affine'),
                 fling_aug_types: tuple = ('depth', 'flip', 'affine', 'random_permute'),
                 depth_scale_range: tuple = (0.8, 1.2),
                 max_depth_offset: float = 0.03,
                 max_normal_rot_angle: float = 20,
                 max_fling_rot_angle: float = 30,
                 label_smoothing_value: float = 0.15,
                 # hyper-params
                 num_pc_sample: int = 8000,
                 num_pc_sample_final: int = 4000,
                 voxel_size: float = 0.002,
                 # dataset
                 primitive_classes: tuple = ('fling', 'drag', 'fold1', 'fold2', 'pick_and_place', 'done'),
                 primitive_idxs: tuple = (0, 1, 2, 3, 4, 5),
                 static_epoch_seed: bool = False,
                 debug: bool = False,
                 num_rankings_per_sample: int = 44,
                 # database config
                 use_database: bool = True,
                 log_api: str = "/v1/logs",
                 log_endpoint: str = "http://100.100.100.105:8080",
                 **kwargs
                 ):
        super().__init__()
        self.logging_dir = logging_dir
        self.namespace = namespace
        self.tag = tag
        self.episode_range = episode_range
        self.num_rankings_per_sample = num_rankings_per_sample

        # database
        self.use_database = use_database
        self.log_api = log_api
        self.log_endpoint = log_endpoint
        # find all data samples
        self.data_samples_list = []
        self.data_samples_path_list = []
        self.find_data_samples()

        # hyper-params
        self.num_pc_sample = num_pc_sample
        self.voxel_size = voxel_size
        self.num_pc_sample_final = num_pc_sample_final
        # data augmentation
        self.label_smoothing_value = label_smoothing_value
        self.use_augmentation = use_augmentation
        self.normal_aug_types = normal_aug_types
        self.fling_aug_types = fling_aug_types
        depth_scale_range = depth_scale_range
        max_depth_offset = max_depth_offset
        max_fling_rot_angle = max_fling_rot_angle
        self.transform_action_normal = None
        self.transform_action_fling = None
        if use_augmentation:
            normal_aug_list = []
            fling_aug_list = []
            if 'depth' in self.normal_aug_types:
                normal_aug_list.append(aug.Depth(scale_range=depth_scale_range, max_offset=max_depth_offset))
            if 'depth' in self.fling_aug_types:
                fling_aug_list.append(aug.Depth(scale_range=depth_scale_range, max_offset=max_depth_offset))
            assert 'flip' not in self.normal_aug_types, 'Do not support flip transforms for normal action!'
            if 'flip' in self.fling_aug_types:
                fling_aug_list.append(aug.Flip(lr_percent=0.5, ud_percent=0.))  # only flip left-right
            if 'affine' in self.normal_aug_types:
                normal_aug_list.append(aug.Affine(
                    x_trans_range=(-0.1, 0.1),
                    y_trans_range=(-0.1, 0.1),
                    rot_angle_range=(-np.pi / 180 * max_normal_rot_angle, np.pi / 180 * max_normal_rot_angle),
                    scale_range=(0.8, 1.2),
                    trans_place_pose=True,
                ))
            if 'affine' in self.fling_aug_types:
                fling_aug_list.append(aug.Affine(
                    x_trans_range=(-0.2, 0.2),
                    y_trans_range=(-0.15, 0.15),
                    rot_angle_range=(-np.pi / 180 * max_fling_rot_angle, np.pi / 180 * max_fling_rot_angle),
                    scale_range=(0.8, 1.2),
                    trans_place_pose=False,
                ))
            assert 'auto_permute' not in self.normal_aug_types, 'Do not support AutoPermutePose for normal actions!'
            if 'auto_permute' in self.fling_aug_types:
                fling_aug_list.append(aug.AutoPermutePose())
            assert 'random_permute' not in self.normal_aug_types, 'Do not support RandomPermutePose for normal actions!'
            if 'random_permute' in self.fling_aug_types:
                fling_aug_list.append(aug.RandomPermutePose())
            self.transform_action_normal = aug.Sequential(normal_aug_list)
            self.transform_action_fling = aug.Sequential(fling_aug_list)

        self.primitive_classes = list(primitive_classes)
        self.primitive_idxs = list(primitive_idxs)
        self.raw_primitive_classes = ('fling', 'drag', 'fold', 'pick_and_place', 'none', 'done')
        # add ratio statistics for different actions

        # dataset
        self.static_epoch_seed = static_epoch_seed
        self.debug = debug

    def find_data_samples(self):
        """Find data samples from log files."""
        if self.use_database:  # use MongoDB database
            log_dir = osp.join(self.logging_dir, self.namespace, self.tag, 'archives')
            logger.info(f'Loading data samples from {log_dir}...')
            session = requests.Session()
            url = self.log_endpoint + self.log_api
            query_filter = {"$and": [
                {f"metadata.{self.namespace}.tag": {"$exists": "true",  "$eq": self.tag}},
                {f"metadata.{self.namespace}.episode_idx": {"$exists": "true", "$gte": self.episode_range[0]}},
                {f"metadata.{self.namespace}.episode_idx": {"$exists": "true", "$lt": self.episode_range[1]}},
                {f"metadata.{self.namespace}.action.type": {"$in": ["fling", "fold1", "fold2"]}},
                {f"metadata.{self.namespace}.pcd": {"$exists": "true"}},
                {f"metadata.{self.namespace}.annotation.grasp_point_rankings": {"$size": self.num_rankings_per_sample}},
                # {"annotators": {"$all": ["nobody"]}}
            ]}
            start_time = time.time()
            response = session.get(url, json={"extra_filter": query_filter})
            response_dict_list = [json.loads(jline) for jline in response.content.splitlines()]
            self.data_samples_list = list(map(lambda x: x["metadata"], response_dict_list))
            self.data_samples_path_list = list(map(lambda x: osp.join(log_dir, x["identifier"]), response_dict_list))
            end_time = time.time()
            logger.debug(f'Use time (s): {end_time - start_time} for filtering and loading data samples!')
            session.close()
        else:
            log_dir = osp.join(self.logging_dir, self.namespace, self.tag)
            logger.debug(f'Loading data samples from {log_dir}...')
            raw_log_files = os.listdir(log_dir)
            log_files = []
            for log_file in raw_log_files:
                annotation_path = osp.join(log_dir, log_file, 'annotation.yaml')
                if osp.exists(annotation_path):
                    log_files.append(log_file)
            log_files.sort()

            logger.debug(f'Find {len(log_files)} possible valid samples!')
            logger.debug('Loading metadata....')

            start_time = time.time()
            with ThreadPoolExecutor(max_workers=24) as executor:
                data_generator = executor.map(lambda i:
                                              (osp.join(log_dir, log_files[i]),
                                               OmegaConf.load(osp.join(log_dir, log_files[i], 'metadata.yaml'))),
                                              range(len(log_files)))
            data_samples_all = list(data_generator)
            # we don't need pick_and_place any more
            filter_func = lambda x: self.namespace in x[1] and 'episode_idx' in x[1][self.namespace] and 'annotation' in x[1][self.namespace] and \
                                    (self.episode_range[0] <= x[1][self.namespace].episode_idx < self.episode_range[1]) and \
                                    x[1][self.namespace].action.type != 'pick_and_place' and \
                                    len(x[1][self.namespace].annotation.grasp_point_rankings) <= 40
            self.data_samples_path_list = list(map(lambda x: x[0], filter(filter_func, data_samples_all)))
            self.data_samples_list = list(map(lambda x: x[1], filter(filter_func, data_samples_all)))
            end_time = time.time()
            logger.debug(f'Use time (s): {end_time - start_time} for filtering data samples!')

        logger.info(f'Found {len(self.data_samples_list)} data samples.')

    def transform_input(self, pts_xyz: np.ndarray, seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rs = np.random.RandomState(seed=seed)
        all_idxs = np.arange(pts_xyz.shape[0])
        # random select fixed number of points
        if all_idxs.shape[0] >= self.num_pc_sample:
            selected_idxs = rs.choice(all_idxs, size=self.num_pc_sample, replace=False)
        else:
            np.random.seed(seed)
            np.random.shuffle(all_idxs)
            res_num = len(all_idxs) - self.num_pc_sample
            selected_idxs = np.concatenate([all_idxs, all_idxs[:res_num]], axis=0)
        pc_xyz_slim = pts_xyz[selected_idxs, :]

        # perform voxelization for Sparse ResUnet-3D
        _, sel_pc_idxs = ME.utils.sparse_quantize(pc_xyz_slim / self.voxel_size, return_index=True)
        origin_slim_pc_num = sel_pc_idxs.shape[0]
        assert origin_slim_pc_num >= self.num_pc_sample_final
        all_idxs = np.arange(origin_slim_pc_num)
        rs = np.random.RandomState(seed=seed)
        final_selected_idxs = rs.choice(all_idxs, size=self.num_pc_sample_final, replace=False)
        sel_pc_idxs = sel_pc_idxs[final_selected_idxs]
        assert sel_pc_idxs.shape[0] == self.num_pc_sample_final
        # voxelized coords for MinkowskiEngine engine
        coords = np.floor(pc_xyz_slim[sel_pc_idxs, :] / self.voxel_size)
        feat = pc_xyz_slim[sel_pc_idxs, :]
        pts_xyz = pc_xyz_slim[sel_pc_idxs, :]
        return pts_xyz, coords, feat

    def get_poses(self, data_sample: OmegaConf):
        raw_poses = np.asarray(data_sample[self.namespace]['annotation']['action_poses'])
        pose = np.stack([raw_poses[0, :3],  # left pick
                         raw_poses[2, :3],  # left place
                         raw_poses[1, :3],  # right pick
                         raw_poses[3, :3]   # right place
                         ]).astype(np.float32)
        # TODO: use real angles
        pose_angles = np.zeros((4, 1), dtype=np.float32)
        pose = np.concatenate([pose, pose_angles], axis=-1)  # (4, 4)
        pose[np.isnan(pose)] = 0.
        return pose

    def get_fling_rankings(self, data_sample: OmegaConf):
        grasp_point_pred = np.asarray(data_sample[self.namespace]['pose_virtual']['prediction']['begin']).astype(np.float32)  # (K, 6)
        grasp_point_all = grasp_point_pred[:, :3]  # (K, 3) we only need (x, y, z) coordinate
        # [p1_idx1, p1_idx2, p2_idx1, p2_idx2]
        selected_grasp_point_indices = \
            np.asarray(data_sample[self.namespace]['annotation']['selected_grasp_point_indices'], dtype=np.int32)  # (P, 4)
        grasp_point_pair1 = np.stack([grasp_point_all[selected_grasp_point_indices[:, 0], :],
                                      grasp_point_all[selected_grasp_point_indices[:, 1], :]],
                                     axis=0)  # (2, P, 3)
        grasp_point_pair2 = np.stack([grasp_point_all[selected_grasp_point_indices[:, 2], :],
                                      grasp_point_all[selected_grasp_point_indices[:, 3], :]],
                                     axis=0)  # (2, P, 3)
        # load rankings
        rankings = np.asarray(data_sample[self.namespace]['annotation']['grasp_point_rankings']).astype(np.int32)  # (P, )
        # 0: P1 > P2 (P1 is better)
        # 1: P1 < P2 (P2 is better)
        # 2: P1 = P2 (equally good)
        # 3: Not comparable (hard to distinguish for humans).
        num_rankings = rankings.shape[0]
        pair_scores = np.zeros((num_rankings, 2))  # (P, 2)
        pair_scores[rankings == 0, 0], pair_scores[rankings == 0, 1] = \
            1.0 - self.label_smoothing_value, self.label_smoothing_value  # P1 is better
        pair_scores[rankings == 1, 0], pair_scores[rankings == 1, 1] = \
            self.label_smoothing_value, 1.0 - self.label_smoothing_value  # P2 is better
        pair_scores[rankings == 2, :] = 0.5  # equally good
        pair_scores[rankings == 3, :] = 0.  # invalid

        return grasp_point_pair1, grasp_point_pair2, pair_scores

    def __len__(self):
        return len(self.data_samples_list)

    def __getitem__(self, index: int) -> Tuple:
        data_sample = self.data_samples_list[index]
        data_sample_path = self.data_samples_path_list[index]
        # load point cloud
        pcd_path = osp.join(data_sample_path, data_sample[self.namespace]['pcd']['processed']['begin'])
        if not osp.exists(pcd_path):
            pcd_path = pcd_path + '.npz'
        if osp.exists(pcd_path):
            pc_xyz = np.load(pcd_path)['points'].astype(np.float32)
        else:
            pcd_path = pcd_path.replace('.npz', '.ply')
            assert osp.exists(pcd_path), f'pcd_path: {pcd_path} does not exist!'
            pcd = o3d.io.read_point_cloud(pcd_path)
            pc_xyz = np.asarray(pcd.points).astype(np.float32)
        # sample point cloud and transform to sparse tensor
        pts_xyz_numpy, coords_numpy, feat_numpy = self.transform_input(pc_xyz, seed=index if self.static_epoch_seed else None)

        action_type = ActionTypeDef(data_sample[self.namespace]['annotation']['action_type'])
        action_str = ActionTypeDef.to_string(action_type)
        action_idx = self.primitive_classes.index(action_str)

        # generate action poses
        poses_numpy = self.get_poses(data_sample)
        # transform action poses (data augmentation)
        if action_type == ActionTypeDef.FLING:
            pts_xyz_numpy, poses_numpy = self.transform_action_fling(pts_xyz_numpy, poses_numpy)
        else:
            pts_xyz_numpy, poses_numpy = self.transform_action_normal(pts_xyz_numpy, poses_numpy)
        # get fling-rankings
        grasp_point_pair1_numpy, grasp_point_pair2_numpy, grasp_pair_scores_numpy = \
            self.get_fling_rankings(data_sample)
        # transform to torch tensor
        coords = torch.from_numpy(coords_numpy)
        feat = torch.from_numpy(feat_numpy)
        pts_xyz_torch = torch.from_numpy(pts_xyz_numpy)
        poses_torch = torch.from_numpy(poses_numpy)
        gripper_points = poses_torch[:, :3]  # (4, 3)
        grasp_point_pair1, grasp_point_pair2, grasp_pair_scores = torch.from_numpy(grasp_point_pair1_numpy), \
            torch.from_numpy(grasp_point_pair2_numpy), torch.from_numpy(grasp_pair_scores_numpy)
        # action score
        primitive_index = torch.tensor(action_idx, dtype=torch.long)
        smoothed_score = torch.tensor(1.0) if action_type == ActionTypeDef.FOLD_1 else torch.tensor(0.)
        # TODO: more flexible
        folding_score = torch.tensor(0.) if (action_type == ActionTypeDef.FOLD_1 or action_type == ActionTypeDef.FOLD_2) \
            else torch.tensor(1.0)

        return coords, feat, pts_xyz_torch, gripper_points, \
            grasp_point_pair1, grasp_point_pair2, grasp_pair_scores, \
            primitive_index, smoothed_score, folding_score

class RuntimeDataModuleReal(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        dataset_split: tuple of (train, val, test)
        """
        super().__init__()
        assert (len(kwargs['dataset_split']) == 3)
        self.kwargs = kwargs

        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        if self.train_dataset is not None and self.test_dataset is not None:
            return
        kwargs = self.kwargs
        split_seed = kwargs['split_seed']
        dataset_split = kwargs['dataset_split']

        train_args = dict(kwargs)
        train_dataset = RuntimeDatasetReal(**train_args)
        val_dataset = copy.copy(train_dataset)
        val_dataset.use_augmentation = False
        val_dataset.static_epoch_seed = True

        data_samples_all = train_dataset.data_samples_list

        # split for train/val/test
        num_samples = len(data_samples_all)
        normalized_split = np.array(dataset_split)
        normalized_split = normalized_split / np.sum(normalized_split)
        sample_split = (normalized_split * num_samples).astype(np.int64)

        # add leftover instance to training set
        sample_split[0] += num_samples - np.sum(sample_split)

        # generate index for each
        all_idxs = np.arange(num_samples)
        rs = np.random.RandomState(seed=split_seed)
        perm_all_idxs = rs.permutation(all_idxs)

        split_sample_idx_list = list()
        prev_idx = 0
        for x in sample_split:
            next_idx = prev_idx + x
            split_sample_idx_list.append(perm_all_idxs[prev_idx: next_idx])
            prev_idx = next_idx
        assert (np.allclose([len(x) for x in split_sample_idx_list], sample_split))

        # generate subsets
        train_idxs, val_idxs, test_idxs = split_sample_idx_list
        train_subset = Subset(train_dataset, train_idxs)
        val_subset = Subset(val_dataset, val_idxs)
        test_subset = Subset(val_dataset, test_idxs)

        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset

        self.use_weighted_sampler = self.kwargs['weighted_sampler']['enable']
        sample_weights_mode = self.kwargs['weighted_sampler']['mode']
        if self.use_weighted_sampler:
            episode_idx_list_all = [data_sample[self.train_dataset.namespace]['episode_idx']
                                for data_sample in self.train_dataset.data_samples_list]
            episode_idx_list_train = [episode_idx_list_all[idx] for idx in self.train_idxs]
            weights_generator = SampleWeightsGenerator(episode_idx_list=episode_idx_list_train,
                                                       mode=sample_weights_mode,
                                                       min_weight=self.kwargs['weighted_sampler']['min_weight'])
            self.sample_weights = weights_generator.weights
        else:
            self.sample_weights = None

    def train_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        train_subset = self.train_subset
        if self.use_weighted_sampler:
            max_sample_num = self.kwargs['weighted_sampler']['max_sample_num']
            sampler = WeightedRandomSampler(weights=self.sample_weights,
                                            num_samples=max_sample_num,
                                            replacement=True)
        else:
            sampler = None
        dataloader = DataLoader(train_subset,
                                batch_size=batch_size,
                                shuffle=not self.use_weighted_sampler,
                                num_workers=num_workers,
                                persistent_workers=False,
                                drop_last=True,
                                collate_fn=self.collate_pair_fn,
                                sampler=sampler)
        return dataloader

    def val_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        val_subset = self.val_subset
        dataloader = DataLoader(val_subset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                drop_last=True,
                                collate_fn=self.collate_pair_fn)
        return dataloader

    def test_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        test_subset = self.test_subset
        dataloader = DataLoader(test_subset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                drop_last=True,
                                collate_fn=self.collate_pair_fn)
        return dataloader

    def collate_pair_fn(self, list_data: list) -> tuple:
        result_list = [[] for _ in range(len(list_data[0]) - 2)]
        coords_list = []
        feat_list = []
        for data in list_data:
            for type_idx, item in enumerate(data):
                if type_idx == 0:
                    coords_list.append(item)
                elif type_idx == 1:
                    feat_list.append(item)
                else:
                    result_list[type_idx - 2].append(item)
        final_list = [torch.stack(data) for data in result_list]
        coords, feat = ME.utils.sparse_collate(coords_list, feat_list)
        return (coords, feat) + tuple(final_list)