from torch.utils.data import Subset, Dataset, DataLoader
import os
import os.path as osp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from common.visualization_util import visualize_pc_and_grasp_points
import torch
import time
import copy
import pytorch_lightning as pl
from loguru import logger
from omegaconf import OmegaConf
from typing import Tuple
import tqdm
import requests
import json
import learning.datasets.augmentation as aug
from torch.utils.data import WeightedRandomSampler
from learning.datasets.weighted_sampler import SampleWeightsGenerator
import MinkowskiEngine as ME
import multiprocessing


def process_json_lines(jlines):
    results = []
    for jline in jlines:
        results.append(OmegaConf.create(json.loads(jline)))
    return results


class RuntimeDatasetVirtual(Dataset):
    """RuntimeDatasetVirtual is a virtual dataset
    that is used to filter and generate data samples at runtime."""

    def __init__(self,
                 # logging params
                 logging_dir: str = './log',
                 namespace: str = 'virtual',
                 tag: str = 'debug',
                 episode_range: tuple = (0, 100),
                 # data augmentation
                 use_augmentation: bool = True,
                 fling_aug_types: tuple = ('depth', 'flip', 'affine', 'random_permute'),
                 depth_scale_range: tuple = (0.2, 1.5),
                 max_depth_offset: float = 0.03,
                 max_fling_rot_angle: float = 90,
                 # hyper-params
                 num_pc_sample: int = 8000,
                 num_pc_sample_final: int = 4000,
                 voxel_size: float = 0.001,
                 # dataset
                 static_epoch_seed: bool = False,
                 debug: bool = False,
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
        self.use_augmentation = use_augmentation
        self.fling_aug_types = fling_aug_types
        depth_scale_range = depth_scale_range
        max_depth_offset = max_depth_offset
        max_fling_rot_angle = max_fling_rot_angle
        self.transform_action_fling = None
        if self.use_augmentation:
            fling_aug_list = []
            if 'depth' in self.fling_aug_types:
                fling_aug_list.append(aug.Depth(scale_range=depth_scale_range, max_offset=max_depth_offset))
            if 'flip' in self.fling_aug_types:
                fling_aug_list.append(aug.Flip(lr_percent=0.5, ud_percent=0.25))
            if 'affine' in self.fling_aug_types:
                fling_aug_list.append(aug.Affine(
                    x_trans_range=(-0.2, 0.2),
                    y_trans_range=(-0.15, 0.15),
                    rot_angle_range=(-np.pi / 180 * max_fling_rot_angle, np.pi / 180 * max_fling_rot_angle),
                    scale_range=(0.8, 1.2),
                    trans_place_pose=False,
                ))
            if 'auto_permute' in self.fling_aug_types:
                fling_aug_list.append(aug.AutoPermutePose())
            if 'random_permute' in self.fling_aug_types:
                fling_aug_list.append(aug.RandomPermutePose())
            self.transform_action_fling = aug.Sequential(fling_aug_list)

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
                {f"metadata.{self.namespace}.tag": {"$exists": "true", "$eq": self.tag}},
                {f"metadata.{self.namespace}.episode_idx": {"$exists": "true", "$gte": self.episode_range[0]}},
                {f"metadata.{self.namespace}.episode_idx": {"$exists": "true", "$lt": self.episode_range[1]}},
            ]}
            start_time = time.time()
            response = session.get(url, json={"extra_filter": query_filter})
            json_lines = response.content.splitlines()

            # Define the number of processes to use
            num_processes = multiprocessing.cpu_count()

            # Split the json_lines into chunks for each process
            chunk_size = len(json_lines) // num_processes
            chunks = [json_lines[i:i + chunk_size] for i in range(0, len(json_lines), chunk_size)]

            # Create a multiprocessing Pool
            pool = multiprocessing.Pool(processes=num_processes)

            # Apply the process_json_line function to each chunk
            results = pool.map(process_json_lines, tqdm.tqdm(chunks))

            # Combine the results
            response_dict_list = []
            for result in results:
                response_dict_list.extend(result)
            pool.close()

            # response_dict_list = []
            # for jline in tqdm.tqdm(json_lines):
            #     response_dict_list.append(OmegaConf.create(json.loads(jline)))
            self.data_samples_list = list(map(lambda x: x.metadata, response_dict_list))
            self.data_samples_path_list = list(map(lambda x: osp.join(log_dir, x.identifier), response_dict_list))
            end_time = time.time()
            logger.debug(f'Use time (s): {end_time - start_time} for filtering and loading data samples!')
            session.close()
        else:
            log_dir = osp.join(self.logging_dir, self.namespace, self.tag)
            log_files = os.listdir(log_dir)
            log_files.sort()

            logger.debug(f'Loading data samples from {log_dir}...')
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=24) as executor:
                data_generator = executor.map(lambda i:
                                              (osp.join(log_dir, log_files[i]),
                                               OmegaConf.load(osp.join(log_dir, log_files[i], 'metadata.yaml'))),
                                              range(len(log_files)))
            data_samples_all = list(data_generator)
            filter_func = lambda x: 'episode_idx' in x[1][self.namespace] and x[1][self.namespace].episode_idx is not None and \
                                    (self.episode_range[0] <= x[1][self.namespace].episode_idx < self.episode_range[1])
            self.data_samples_path_list = list(map(lambda x: x[0], filter(filter_func, data_samples_all)))
            self.data_samples_list = list(map(lambda x: x[1], filter(filter_func, data_samples_all)))
            end_time = time.time()
            logger.debug(f'Use time (s): {end_time - start_time} for filtering data samples!')

        logger.info(f'Found {len(self.data_samples_list)} data samples.')

    def transform_input(self,
                        pts_xyz: np.ndarray,
                        pts_nocs: np.ndarray,
                        seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        pc_nocs_slim = pts_nocs[selected_idxs, :]
        assert pc_xyz_slim.shape[0] == pc_nocs_slim.shape[0]

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
        pts_nocs = pc_nocs_slim[sel_pc_idxs, :]
        return pts_xyz, coords, feat, pts_nocs

    @staticmethod
    def generate_poses_from_grasp_points(left_grasp_point: np.ndarray, right_grasp_point: np.ndarray) -> np.ndarray:
        # TODO: support place points
        poses = np.stack([left_grasp_point,
                          np.zeros_like(left_grasp_point), # left place point
                          right_grasp_point,
                          np.zeros_like(right_grasp_point) # right place point
                          ], axis=0).astype(np.float32)
        # TODO: add correct rotation angle (theta)
        poses = np.concatenate([poses, np.zeros((4, 1), dtype=np.float32)], axis=1)  # (4, 4)
        return poses

    @staticmethod
    def get_delta_deformable_reward(reward_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        begin_reward_dict = reward_dict['begin']
        end_reward_dict = reward_dict['end']
        delta_reward_dict = dict()
        for key in begin_reward_dict.keys():
            if key in ('l2_distance', 'icp_distance'):
                # the smaller distance the better reward
                delta_reward_dict[key] = begin_reward_dict[key] - end_reward_dict[key]
        canonicalization_reward, alignment_reward = delta_reward_dict['l2_distance'], delta_reward_dict['icp_distance']
        return torch.tensor(canonicalization_reward), torch.tensor(alignment_reward)

    @staticmethod
    def find_grasp_point_nocs(pc_xyz: torch.Tensor, pc_nocs: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        num_pts = pc_xyz.shape[0]
        pc_xyz_expand = pc_xyz.view(-1, 1, 3).expand(-1, 2, -1)  # (N, 2, 3)
        grasp_points = poses[0:3:2, :3].view(1, -1, 3).expand(num_pts, -1, -1)  # (N, 2, 3)
        dist = torch.square(pc_xyz_expand - grasp_points).sum(dim=-1)  # (N, 2)
        min_dixs = torch.argmin(dist, dim=0)  # (2, )
        grasp_pts_nocs = pc_nocs[min_dixs, :]  # (2, 3)
        return grasp_pts_nocs

    def __len__(self):
        return len(self.data_samples_list)

    def __getitem__(self, index: int) -> Tuple:
        data_sample = self.data_samples_list[index]
        data_sample_path = self.data_samples_path_list[index]
        # load point cloud nocs
        nocs_path = osp.join(data_sample_path, data_sample[self.namespace].pcd.processed_nocs.begin)
        assert osp.exists(nocs_path), f'nocs_path: {nocs_path} does not exist!'
        pc_nocs = np.load(nocs_path)['nocs'].astype(np.float32)
        # load point cloud xyz
        pcd_path = osp.join(data_sample_path, data_sample[self.namespace].pcd.processed.begin)
        if not osp.exists(pcd_path):
            pcd_path = pcd_path + '.npz'
        assert osp.exists(pcd_path), f'pcd_path: {pcd_path} does not exist!'
        pc_xyz = np.load(pcd_path)['points'].astype(np.float32)
        # sample point cloud and transform to sparse tensor
        pts_xyz_numpy, coords_numpy, feat_numpy, pts_nocs_numpy = \
            self.transform_input(pc_xyz,
                                 pc_nocs,
                                 seed=index if self.static_epoch_seed else None)

        # load action poses
        is_best_action = data_sample[self.namespace].pose_virtual.is_best
        if is_best_action is None:
            is_best_action = False
        is_best_action = np.array(is_best_action, dtype=np.float32)
        left_grasp_point = np.asarray(data_sample[self.namespace]
                                      .pose_virtual.gripper.begin.left)[:3].astype(np.float32)
        right_grasp_point = np.asarray(data_sample[self.namespace]
                                       .pose_virtual.gripper.begin.right)[:3].astype(np.float32)
        # generate action poses
        poses_numpy = self.generate_poses_from_grasp_points(left_grasp_point, right_grasp_point)
        # transform action poses (data augmentation)
        if self.use_augmentation:
            pts_xyz_numpy, poses_numpy = self.transform_action_fling(pts_xyz_numpy, poses_numpy)

        # transform to torch tensor
        coords = torch.from_numpy(coords_numpy)
        feat = torch.from_numpy(feat_numpy)
        pts_xyz_torch = torch.from_numpy(pts_xyz_numpy)
        pts_nocs_torch = torch.from_numpy(pts_nocs_numpy)
        valid_poses_weight = torch.from_numpy(is_best_action)
        poses_torch = torch.from_numpy(poses_numpy)
        gripper_points = poses_torch[:, :3]  # (4, 3)

        # find grasp points nocs
        grasp_points_nocs_torch = self.find_grasp_point_nocs(pts_xyz_torch, pts_nocs_torch, poses_torch)  # (2, 3)
        if self.debug:
            logger.debug(
                f'Cloth Dynamics Data: max height difference: {np.max(pts_xyz_numpy[:, 2]) - np.min(pts_xyz_numpy[:, 2])}')
            left_pick_point = poses_numpy[0, :3]
            right_pick_point = poses_numpy[2, :3]
            visualize_pc_and_grasp_points(pts_xyz_numpy, left_pick_point, right_pick_point, pc_colors=pts_nocs_numpy)
        # get reward
        raw_reward_dict = dict(data_sample[self.namespace].reward)
        canonicalization_reward, alignment_reward = self.get_delta_deformable_reward(raw_reward_dict)

        return coords, feat, pts_xyz_torch, pts_nocs_torch, gripper_points, grasp_points_nocs_torch, valid_poses_weight, \
            canonicalization_reward, alignment_reward

class RuntimeDataModuleVirtual(pl.LightningDataModule):
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
        train_dataset = RuntimeDatasetVirtual(**train_args)
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