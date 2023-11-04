import os
import os.path as osp
import open3d as o3d
import copy
import numpy as np
import pandas as pd
import torch
import tqdm
import time
from typing import Tuple
from loguru import logger
import bson  # pip install pymongo
import learning.datasets.augmentation as aug
import pytorch_lightning as pl
from torch.utils.data import Subset, Dataset, DataLoader
import MinkowskiEngine as ME


# data sets
# =========
class VirtualRealityDataset(Dataset):
    """
    Dataset for virtual reality data
    """
    def __init__(self,
                 # h5 path
                 h5_path: str,
                 # data augumentaiton
                 use_augmentation: bool = False,
                 normal_aug_types: tuple = ('depth', 'affine'),
                 fling_aug_types: tuple = ('depth', 'flip', 'affine', 'random_permute'),
                 fold_aug_types: tuple = ('depth', 'affine'),
                 depth_scale_range: tuple = (0.2, 1.5),
                 max_depth_offset: float = 0.03,
                 max_normal_rot_angle: float = 20,
                 max_fold_rot_angle: float = 20,
                 max_fling_rot_angle: float = 180,
                 # hyper-params
                 num_pc_sample: int = 8000,
                 num_pc_sample_final: int = 4000,
                 num_rotations: int = 10,
                 sigma: float = 0.025,
                 delta_coverage_as_reward=True,
                 use_tanh_reward=False,
                 reward_alpha: float = 2.0,
                 reward_beta: float = 1.0,
                 voxel_size: float = 0.001,
                 # dataset
                 primitive_classes: tuple = ('fling', 'drag', 'fold1', 'fold2', 'pick_and_place', 'done'),
                 primitive_idxs: tuple = (0, 1, 2, 3, 4, 5),
                 static_epoch_seed: bool = False,
                 # debug
                 debug: bool = False,
                 **kwargs):

        super().__init__()
        self.data_dir = osp.dirname(h5_path)
        assert osp.exists(h5_path)
        with pd.HDFStore(h5_path) as store:
            self.df = store['df']
        # data augmentation
        self.use_augmentation = use_augmentation
        self.normal_aug_types = normal_aug_types
        self.fling_aug_types = fling_aug_types
        self.fold_aug_types = fold_aug_types
        # debug config
        self.debug = debug

        # hyper-parms
        self.num_pc_sample = num_pc_sample
        self.num_pc_sample_final = num_pc_sample_final
        self.num_rotations = num_rotations
        self.sigma = sigma
        self.delta_coverage_as_reward = delta_coverage_as_reward
        self.use_tanh_reward = use_tanh_reward
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.static_epoch_seed = static_epoch_seed
        self.voxel_size = voxel_size

        self.transform_action_normal = None
        self.transform_action_fling = None
        if use_augmentation:
            normal_aug_list = []
            fling_aug_list = []
            fold_aug_list = []
            if 'depth' in self.normal_aug_types:
                normal_aug_list.append(aug.Depth(scale_range=depth_scale_range, max_offset=max_depth_offset))
            if 'depth' in self.fling_aug_types:
                fling_aug_list.append(aug.Depth(scale_range=depth_scale_range, max_offset=max_depth_offset))
            if 'depth' in self.fold_aug_types:
                fold_aug_list.append(aug.Depth(scale_range=depth_scale_range, max_offset=max_depth_offset))
            assert 'flip' not in self.normal_aug_types, 'Do not support flip transforms for normal action!'
            assert 'flip' not in self.fold_aug_types, 'Do not support flip transforms for fold action!'
            if 'flip' in self.fling_aug_types:
                fling_aug_list.append(aug.Flip(lr_percent=0.5, ud_percent=0.25))
            if 'affine' in self.normal_aug_types:
                normal_aug_list.append(aug.Affine(
                    x_trans_range=(-0.2, 0.2),
                    y_trans_range=(-0.15, 0.15),
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
            if 'affine' in self.fold_aug_types:
                fold_aug_list.append(aug.Affine(
                    x_trans_range=(-0.2, 0.2),
                    y_trans_range=(-0.15, 0.15),
                    rot_angle_range=(-np.pi / 180 * max_fold_rot_angle, np.pi / 180 * max_fold_rot_angle),
                    scale_range=(0.8, 1.2),
                    trans_place_pose=True,
                ))

            assert 'auto_permute' not in self.normal_aug_types, 'Do not support AutoPermutePose for normal actions!'
            assert 'auto_permute' not in self.fold_aug_types, 'Do not support AutoPermutePose for fold actions!'
            if 'auto_permute' in self.fling_aug_types:
                fling_aug_list.append(aug.AutoPermutePose())
            assert 'random_permute' not in self.normal_aug_types, 'Do not support RandomPermutePose for normal actions!'
            assert 'random_permute' not in self.fold_aug_types, 'Do not support RandomPermutePose for fold actions!'
            if 'random_permute' in self.fling_aug_types:
                fling_aug_list.append(aug.RandomPermutePose())
            self.transform_action_normal = aug.Sequential(normal_aug_list)
            self.transform_action_fling = aug.Sequential(fling_aug_list)
            self.transform_action_fold = aug.Sequential(fold_aug_list)

        self.primitive_classes = list(primitive_classes)
        self.primitive_idxs = list(primitive_idxs)
        self.raw_primitive_classes = ('fling', 'drag', 'fold', 'pick_and_place', 'none', 'done')
        total_counts = {self.raw_primitive_classes[idx]: len(list(filter(lambda a: a == idx, self.df['action_type'])))
                        for idx in range(len(self.raw_primitive_classes))}
        print(f'Init dataset with {len(self.df)} actions')
        print(f"Count ratios for " + ', '.join([f'{k}: {v/len(self.df):0.3f}' for k, v in total_counts.items()]))
        print(f"Balance factor all fling: {1.0:0.3f}, drag: {total_counts['drag']/total_counts['fold']:0.3f}, fold: {total_counts['fling']/total_counts['fold']:0.3f}, done: {total_counts['fling']/total_counts['done']:0.3f}")

    def __len__(self):
        return len(self.df)

    def find_max_coverage(self, idx: int) -> float:
        instance_id = self.df['instance_id'][idx]
        cur_instance_frames = self.df[self.df['instance_id'] == instance_id]
        max_coverage = cur_instance_frames['coverage_frame1'].max()
        return max_coverage

    def calc_reward(self, idx: int) -> float:
        max_coverage = self.find_max_coverage(idx)
        normalized_coverage1 = self.df['coverage_frame1'][idx] / max_coverage
        normalized_coverage2 = self.df['coverage_frame2'][idx] / max_coverage
        if self.use_tanh_reward:
            reward = max(np.tanh(self.reward_alpha * (normalized_coverage2 - normalized_coverage1)
                                 + self.reward_beta * normalized_coverage2), 0.)
        else:
            reward = max(normalized_coverage2 - normalized_coverage1, 0.)
        return reward

    def find_fling_action_step(self, idx: int) -> Tuple[int, int]:
        video_id = self.df['video_id'][idx]
        cur_video_frames = self.df[self.df['video_id'] == video_id]
        sample_id = self.df['sample_id'][idx]
        cur_fling_frames = cur_video_frames[cur_video_frames['action_type'] == 0]
        if sample_id in list(cur_fling_frames['sample_id']):
            cur_fling_step = list(cur_fling_frames['sample_id']).index(sample_id)
        else:
            cur_fling_step = None
        max_fling_steps = len(cur_fling_frames)
        return cur_fling_step, max_fling_steps

    def read_pcd(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        pcd_path = osp.join(self.data_dir, self.df['pcd_path_frame1'][idx]).replace('\\', '/')
        assert osp.exists(pcd_path)
        pcd = np.load(pcd_path)
        pc_xyz = pcd[:, :3].astype(np.float32)
        pc_rgb = pcd[:, 3:6].astype(np.float32)
        return pc_rgb, pc_xyz

    def sample_pcd(self, pc_rgb: np.ndarray, pc_xyz: np.ndarray, pc_nocs: np.ndarray, seed=None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert pc_rgb.shape[0] == pc_xyz.shape[0] == pc_nocs.shape[0]
        rs = np.random.RandomState(seed=seed)
        all_idxs = np.arange(pc_rgb.shape[0])
        if all_idxs.shape[0] >= self.num_pc_sample:
            selected_idxs = rs.choice(all_idxs, size=self.num_pc_sample, replace=False)
        else:
            np.random.seed(seed)
            np.random.shuffle(all_idxs)
            res_num = len(all_idxs) - self.num_pc_sample
            selected_idxs = np.concatenate([all_idxs, all_idxs[:res_num]], axis=0)
        return pc_rgb[selected_idxs, :], pc_xyz[selected_idxs, :], pc_nocs[selected_idxs, :]

    def read_nocs(self, idx: int) -> np.ndarray:
        nocs_npy_path = osp.join(self.data_dir, self.df['nocs_path_frame1'][idx]).replace('\\', '/')
        assert osp.exists(nocs_npy_path)
        pc_nocs = np.load(nocs_npy_path).astype(np.float32)
        return pc_nocs

    def gauss_3d_batch(self, pc_xyz: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        pts_dist = pc_xyz - pose[:3].unsqueeze(0)
        pts_score = torch.exp(-(pts_dist * pts_dist).sum(-1) / (2 * self.sigma ** 2))
        return pts_score

    def pose_to_array(self, index: int) -> np.ndarray:
        pose = np.stack([
            np.concatenate([self.df['left_gripper_point_frame1'][index].astype(np.float32),
                            np.array(self.df['left_theta_frame1'][index].astype(np.float32))[np.newaxis]], axis=-1),
            np.concatenate([self.df['left_gripper_point_frame2'][index].astype(np.float32),
                            np.array(self.df['left_theta_frame2'][index]).astype(np.float32)[np.newaxis]], axis=-1),
            np.concatenate([self.df['right_gripper_point_frame1'][index].astype(np.float32),
                            np.array(self.df['right_theta_frame1'][index]).astype(np.float32)[np.newaxis]], axis=-1),
            np.concatenate([self.df['right_gripper_point_frame2'][index].astype(np.float32),
                            np.array(self.df['right_theta_frame2'][index]).astype(np.float32)[np.newaxis]], axis=-1)
        ])
        # TODO: find a better way to filter out NaNs
        pose[np.isnan(pose)] = 0.
        # clip angle range in [-pi, pi]
        idxs = pose[:, -1] > np.pi
        pose[idxs, -1] = pose[idxs, -1] - 2 * np.pi
        idxs = pose[:, -1] < -np.pi
        pose[idxs, -1] = pose[idxs, -1] + 2 * np.pi
        return pose

    def voxelize_pc_points(self, pc_xyz: np.ndarray, pc_nocs: np.ndarray, seed: int = None):
        _, sel_pc_idxs = ME.utils.sparse_quantize(pc_xyz / self.voxel_size, return_index=True)
        origin_slim_pc_num = sel_pc_idxs.shape[0]
        assert origin_slim_pc_num >= self.num_pc_sample_final
        all_idxs = np.arange(origin_slim_pc_num)
        rs = np.random.RandomState(seed=seed)
        final_selected_idxs = rs.choice(all_idxs, size=self.num_pc_sample_final, replace=False)
        sel_pc_idxs = sel_pc_idxs[final_selected_idxs]
        assert sel_pc_idxs.shape[0] == self.num_pc_sample_final
        # voxelized coords for MinkowskiEngine engine
        coords = np.floor(pc_xyz[sel_pc_idxs, :] / self.voxel_size)
        # feat = np.zeros_like(coords)
        feat = pc_xyz[sel_pc_idxs, :]
        return pc_xyz[sel_pc_idxs, :], pc_nocs[sel_pc_idxs, :], coords, feat

    def find_grasp_point_nocs(self, pc_xyz: torch.Tensor, pc_nocs: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        num_pts = pc_xyz.shape[0]
        pc_xyz_expand = pc_xyz.view(-1, 1, 3).expand(-1, 2, -1)  # (N, 2, 3)
        grasp_points = poses[0:3:2, :3].view(1, -1, 3).expand(num_pts, -1, -1)  # (N, 2, 3)
        dist = torch.square(pc_xyz_expand - grasp_points).sum(dim=-1)  # (N, 2)
        min_dixs = torch.argmin(dist, dim=0)  # (2, )
        grasp_pts_nocs = pc_nocs[min_dixs, :]  # (2, 3)
        return grasp_pts_nocs

    def __getitem__(self, index: int):
        action = self.df.loc[index, :]
        action_idx = action['action_type']
        if action_idx == 2 and action['folding_step'] == 1:
            action_idx = 2  # fold1 2 -> 2
        elif action_idx == 2 and action['folding_step'] == 2:
            action_idx = 3  # fold2 2 -> 3
        elif action_idx == 3:
            action_idx = 4  # pick-and-place 3->4
        assert action_idx in self.primitive_idxs, "Unknown action type"
        if not self.df['is_smoothing'][index] and action_idx == 4:  # pick_and_place shouldn't occur in folding actions
            # TODO: fix errors in data
            action_idx = 5  # done
        action_type = self.primitive_classes[action_idx]

        pc_rgb, pc_xyz = self.read_pcd(index)
        pc_nocs = self.read_nocs(index)
        pc_rgb, pc_xyz, pc_nocs = self.sample_pcd(pc_rgb, pc_xyz, pc_nocs, seed=index if self.static_epoch_seed else None)
        poses = self.pose_to_array(index)

        # data augmentation
        if self.transform_action_normal:
            if action_type == 'fling':
                pc_xyz, poses = self.transform_action_fling(pc_xyz, poses)
            elif action_type == 'fold1' or action_type == 'fold2':
                pc_xyz, poses = self.transform_action_fold(pc_xyz, poses)
            else:
                pc_xyz, poses = self.transform_action_normal(pc_xyz, poses)

        poses = torch.from_numpy(poses).float()  # (4, 4)
        pc_xyz, pc_nocs, coords, feat = self.voxelize_pc_points(pc_xyz, pc_nocs)
        coords = torch.from_numpy(coords)
        feat = torch.from_numpy(feat)
        pc_xyz = torch.from_numpy(pc_xyz).float()
        pc_nocs = torch.from_numpy(pc_nocs).float()

        if self.debug:
            logger.debug(f'VR Data: max height difference: {torch.max(pc_xyz[:, 2]) - torch.min(pc_xyz[:, 2])}')
            self.plot_single(pc_xyz.cpu().numpy(), pc_nocs.cpu().numpy(),
                             save_dir='outputs/debug', gt_action=action_type, gt_poses=poses.cpu().numpy(),
                             pcd_id=index, vis=False)

        gripper_points = poses[:, :3]  # (4, 3)
        grasp_points_nocs = self.find_grasp_point_nocs(pc_xyz, pc_nocs, poses)  # (2, 3)

        clip_angles = poses[:, -1] + np.pi / self.num_rotations + np.pi  # make cls at the center of the interval
        clip_angles[clip_angles > np.pi] -= 2 * np.pi  # clip to [0, 2pi]
        rotation_cls = (clip_angles / (2 * np.pi) * self.num_rotations).to(torch.long)
        if action_type == 'done':
            reward = 0.0
        elif action_type == 'fling':
            reward = self.calc_reward(index)
        else:
            reward = 0.8
        reward = torch.tensor(reward, dtype=torch.float32)
        primitive_index = torch.tensor(action_idx, dtype=torch.long)
        smoothed_score = torch.tensor(1.0) if self.df['is_smoothed'][index] else torch.tensor(0.)
        folding_score = torch.tensor(0.) if self.df['is_smoothing'][index] else torch.tensor(1.0)
        folding_step = torch.tensor(self.df['folding_step'][index])

        return coords, feat, pc_xyz, pc_nocs, gripper_points, grasp_points_nocs, rotation_cls, \
               reward, primitive_index, smoothed_score, folding_score, folding_step

    def plot_single(self, pc_xyz, pc_nocs, save_dir, gt_action, gt_poses, pcd_id=0, vis=False):
        if vis:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_nocs)
            pcd.colors = o3d.utility.Vector3dVector(pc_nocs)
            o3d.visualization.draw_geometries([pcd, coord])
        else:
            os.makedirs(os.path.join(save_dir, 'gt_vis_aug' if self.use_augmentation else 'gt_vis'), exist_ok=True)
            data_dict = dict(pc_xyz=pc_xyz.tolist(),
                             pc_nocs=pc_nocs.tolist(),
                             action=gt_action,
                             left_gripper_point_frame1=gt_poses[0, :3].tolist(),
                             left_gripper_point_frame2=gt_poses[1, :3].tolist(),
                             right_gripper_point_frame1=gt_poses[2, :3].tolist(),
                             right_gripper_point_frame2=gt_poses[3, :3].tolist(),
                             left_theta_frame1=gt_poses[0, 3].tolist(),
                             left_theta_frame2=gt_poses[1, 3].tolist(),
                             right_theta_frame1=gt_poses[2, 3].tolist(),
                             right_theta_frame2=gt_poses[3, 3].tolist()
                             )
            data = bson.BSON.encode(data_dict)
            out_path = osp.join(save_dir, 'gt_vis_aug' if self.use_augmentation else 'gt_vis', '{}.bson'.format(pcd_id))
            with open(out_path, 'wb') as f:
                f.write(data)
            print('Saving to {}!'.format(out_path))


# data modules
# ============
class VirtualRealityDataModule(pl.LightningDataModule):
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
        kwargs = self.kwargs
        split_seed = kwargs['split_seed']
        dataset_split = kwargs['dataset_split']

        train_args = dict(kwargs)
        train_dataset = VirtualRealityDataset(**train_args)
        val_dataset = copy.deepcopy(train_dataset)
        val_dataset.use_augmentation = False
        val_dataset.static_epoch_seed = True

        groups_df = train_dataset.df
        instances_df = groups_df.groupby('instance_id').agg({'sample_id': lambda x: sorted(x)})

        # split for train/val/test
        num_instances = len(instances_df)
        normalized_split = np.array(dataset_split)
        normalized_split = normalized_split / np.sum(normalized_split)
        instance_split = (normalized_split * num_instances).astype(np.int64)

        # add leftover instance to training set
        instance_split[0] += num_instances - np.sum(instance_split)

        # generate index for each
        all_idxs = np.arange(num_instances)
        rs = np.random.RandomState(seed=split_seed)
        perm_all_idxs = rs.permutation(all_idxs)

        split_instance_idx_list = list()
        prev_idx = 0
        for x in instance_split:
            next_idx = prev_idx + x
            split_instance_idx_list.append(perm_all_idxs[prev_idx: next_idx])
            prev_idx = next_idx
        assert (np.allclose([len(x) for x in split_instance_idx_list], instance_split))

        split_idx_list = list()
        split_instance_finenames = ['train_instances.txt', 'val_instances.txt', 'test_instances.txt']
        for idx, instance_idxs in enumerate(split_instance_idx_list):
            sorted_instance_id_list = sorted(np.asarray(instances_df.iloc[instance_idxs].index).tolist())
            split_instance_filepath = os.path.join(train_dataset.data_dir, split_instance_finenames[idx])
            if not os.path.exists(split_instance_filepath):
                with open(split_instance_filepath, 'w') as f:
                    for instance_id in sorted_instance_id_list:
                        f.write(instance_id + '\n')
                print('Saving instance split information to {}!'.format(split_instance_filepath))
            else:
                print('Found instance split information from {}!'.format(split_instance_filepath))
            idxs = np.sort(np.concatenate(instances_df.iloc[instance_idxs].sample_id).astype(np.int32))
            split_idx_list.append(idxs)

        # generate subsets
        train_idxs, val_idxs, test_idxs = split_idx_list
        train_subset = Subset(train_dataset, train_idxs)
        val_subset = Subset(val_dataset, val_idxs)
        test_subset = Subset(val_dataset, test_idxs)

        self.groups_df = groups_df
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset

    def train_dataloader(self):
        kwargs = self.kwargs
        batch_size = kwargs['batch_size']
        num_workers = kwargs['num_workers']
        train_subset = self.train_subset
        dataloader = DataLoader(train_subset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                persistent_workers=False,
                                drop_last=True,
                                collate_fn=self.collate_pair_fn)
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


if __name__ == '__main__':
    np.random.seed(242134)
    dataset = VirtualRealityDataset(
        h5_path='/home/hanxue/data/UniFolding/VR_Imitation/vr_imitation_flattening-folding-long_v5_dataset/Tshirt/data.h5',
        use_augmentation=False,
        fling_aug_types=('depth', 'flip', 'affine', 'random_permute'),
        fold_aug_types=('depth', 'affine'),
        normal_aug_types=('depth', 'affine'),
        max_fold_rot_angle=180,
        max_fling_rot_angle=180,
        voxel_size=0.002,
        num_pc_sample_final=4000,
        debug=True
    )
    # length = len(dataset)
    length = 100
    dist_list = []
    for i in tqdm.tqdm(range(length)):
        data = dataset[i]