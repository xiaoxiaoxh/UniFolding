import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class Sequential:
    def __init__(self, augmentors):
        self.augmentors = augmentors

    def __call__(self, points, poses):
        for a in self.augmentors:
            points, poses = a(points, poses)
        return points, poses


class Affine:
    def __init__(self,
                 x_trans_range=None,
                 y_trans_range=None,
                 rot_angle_range=None,
                 scale_range=None,
                 trans_place_pose=False):
        self.x_trans_range = x_trans_range
        self.y_trans_range = y_trans_range
        self.rot_angle_range = rot_angle_range
        self.scale_range = scale_range
        self.trans_place_pose = trans_place_pose  # whether to transform place point in frame 2

    @staticmethod
    def rand_uniform(low=0.0, high=1.0):
        return low + torch.rand(1)[0].numpy() * (high - low)

    def __call__(self, points, poses):
        center = ((np.max(points, axis=0) + np.min(points, axis=0)) / 2)[np.newaxis, :]

        rot_angle = self.rand_uniform(low=self.rot_angle_range[0], high=self.rot_angle_range[1]) if self.rot_angle_range else 0.0
        x_trans = self.rand_uniform(low=self.x_trans_range[0], high=self.x_trans_range[1]) if self.x_trans_range else 0.0
        y_trans = self.rand_uniform(low=self.y_trans_range[0], high=self.y_trans_range[1]) if self.y_trans_range else 0.0
        offset_trans = np.array([[x_trans, y_trans, 0.]]).astype(np.float32)
        scale_trans = self.rand_uniform(low=self.scale_range[0], high=self.scale_range[1]) if self.scale_range else 1.0

        rot_mat = R.from_euler(
            'z', rot_angle, degrees=False
        ).as_matrix().astype(np.float32)

        points = ((points - center) * scale_trans) @ rot_mat.T + center + offset_trans
        for idx in (0, 1, 2, 3) if self.trans_place_pose else (0, 2):
            poses[idx, :3] = ((poses[idx, :3][np.newaxis, :] - center) * scale_trans) @ rot_mat.T + center + offset_trans
            poses[idx, -1] = poses[idx, -1] - rot_angle / 180.0 * np.pi

        # clip angle range in [-pi, pi]
        idxs = poses[:, -1] > np.pi
        poses[idxs, -1] = poses[idxs, -1] - 2 * np.pi
        idxs = poses[:, -1] < -np.pi
        poses[idxs, -1] = poses[idxs, -1] + 2 * np.pi
        return points, poses


class AutoPermutePose:
    def __init__(self):
        pass

    def __call__(self, points, poses):
        if poses[0, 0] > poses[2, 0]:  # x1 > x2, frame 1
            # permute lef-right pose based on x-coordinate
            poses[0, :], poses[2, :] = poses[2, :].copy(), poses[0, :].copy()
        if poses[1, 0] > poses[3, 0]:  # x1 > x2, frame 2
            # permute lef-right pose based on x-coordinate
            poses[1, :], poses[3, :] = poses[3, :].copy(), poses[1, :].copy()
        return points, poses


class RandomPermutePose:
    def __init__(self):
        pass

    def __call__(self, points, poses):
        p1 = torch.rand(1)
        if p1 < 0.5:  # frame 1
            # permute lef-right pick pose
            poses[0, :], poses[2, :] = poses[2, :].copy(), poses[0, :].copy()
        return points, poses


class Flip:
    def __init__(self, lr_percent=None, ud_percent=None):
        self.lr_percent = lr_percent
        self.ud_percent = ud_percent

    def __call__(self, points, poses):
        center = ((np.max(points, axis=0) + np.min(points, axis=0)) / 2)[np.newaxis, :]
        p_ud, p_lr = torch.rand(2)

        if p_ud < self.ud_percent:  # UD
            points[:, 1] = 2 * center[:, 1] - points[:, 1]  # y-axis
            poses[:, 1] = 2 * center[:, 1] - poses[:, 1]
            for idx in (0, 2):
                # only flip frame 1 (grasp point), nor frame 2 (place point)
                poses[idx, -1] = -poses[idx, -1]  # change theta

        if p_lr < self.lr_percent:  # LR
            points[:, 0] = 2 * center[:, 0] - points[:, 0]  # x-axis
            poses[:, 0] = 2 * center[:, 0] - poses[:, 0]  # x-axis
            for idx in (0, 2):
                # only flip frame 1 (grasp point), nor frame 2 (place point)
                poses[idx, -1] = np.pi - poses[idx, -1]  # change theta

        # clip angle range in [-pi, pi]
        idxs = poses[:, -1] > np.pi
        poses[idxs, -1] = poses[idxs, -1] - 2 * np.pi
        idxs = poses[:, -1] < -np.pi
        poses[idxs, -1] = poses[idxs, -1] + 2 * np.pi
        return points, poses


class Depth:
    def __init__(self, scale_range=(0.2, 1.2), max_offset=0.05):
        self.scale_range = scale_range
        self.max_offset = max_offset

    @staticmethod
    def rand_uniform(low=0.0, high=1.0):
        return low + torch.rand(1)[0].numpy() * (high - low)

    def __call__(self, points, poses):
        scale = self.rand_uniform(self.scale_range[0], self.scale_range[1])
        # TODO: support points with non-zero plane height
        points[:, 2] = points[:, 2] * scale  # z-axis
        poses[:, 2] = poses[:, 2] * scale
        offset = self.rand_uniform(0., self.max_offset)
        points[:, 2] = points[:, 2] + offset
        poses[:, 2] = poses[:, 2] + offset
        return points, poses
