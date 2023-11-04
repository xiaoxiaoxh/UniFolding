import dataclasses
import json
from abc import ABC, abstractmethod
from os import path as osp
from typing import Tuple, Dict, Optional, Union

import numpy as np
import open3d as o3d
from autolab_core import RigidTransform, transformations as tr
from easydict import EasyDict
from omegaconf import OmegaConf, DictConfig, ListConfig, open_dict
from scipy.spatial.transform import Rotation

from common.datamodels import ActionTypeDef, GarmentTypeDef, ActionMessage, ObservationMessage, ExceptionMessage


class ExperimentBase(ABC):
    option: OmegaConf

    @abstractmethod
    def __del__(self):
        pass

    @staticmethod
    def assign_tcp_frame(pose_left: RigidTransform, pose_right: RigidTransform):
        if isinstance(pose_left, tuple):
            for p in pose_left:
                p.from_frame = 'l_tcp'
        elif pose_left is not None:
            pose_left.from_frame = 'l_tcp'

        if isinstance(pose_right, tuple):
            for p in pose_right:
                p.from_frame = 'r_tcp'
        elif pose_right is not None:
            pose_right.from_frame = 'r_tcp'

    @abstractmethod
    def assign_to_arm(self, pose1: RigidTransform, pose2: RigidTransform) -> Tuple[RigidTransform, RigidTransform]:
        """returns tuple with (left arm, right arm)"""
        pass

    @abstractmethod
    def is_pose_safe(self, pose1: RigidTransform, pose2: RigidTransform) -> bool:
        """Judge whether the pick pose is safe for the robot to execute, return True if safe, return False if not safe"""
        pass

    @abstractmethod
    def is_pose_within_workspace(self, pose: RigidTransform) -> bool:
        """Judge whether the pick pose is within the workspace, return True if within, return False if not within"""
        pass

    @abstractmethod
    def is_pose_reachable_by_dual_arm(self, pose: RigidTransform) -> Tuple[bool, bool]:
        """Judge whether the pick pose is reachable for the left robot and the right robot"""
        pass

    @abstractmethod
    def is_action_executable(self, action_type: ActionTypeDef,
                             poses: Tuple[RigidTransform, RigidTransform, RigidTransform, RigidTransform],
                             **kwargs) -> Tuple[Optional[Dict[str, RigidTransform]], Optional[ExceptionMessage]]:
        """
        Jude whether the input action with 6D poses is executable,
        return (Dict, None) with transforms (3D poses) if action is valid,
        return (None, ExceptionMessage) if action is not executable
        """
        pass

    @abstractmethod
    def capture_pcd(self) -> Tuple[ObservationMessage, Optional[Exception]]:
        pass

    @abstractmethod
    def execute_action(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        pass


@dataclasses.dataclass
class ExperimentRealTransforms:
    option: DictConfig = dataclasses.field(default=None)
    world_to_camera_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    camera_to_world_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    world_to_left_robot_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    world_to_right_robot_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    left_robot_to_world_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    right_robot_to_world_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    left_robot_base_pos: np.ndarray = dataclasses.field(default=np.array([0.0, 0.0, 0.0]))
    right_robot_base_pos: np.ndarray = dataclasses.field(default=np.array([0.0, 0.0, 0.0]))
    virtual_to_world_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    world_to_virtual_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    virtual_to_camera_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    camera_to_virtual_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    virtual_intrinsic: o3d.camera.PinholeCameraIntrinsic = dataclasses.field(default=None)
    virtual_to_clothfunnels_camera_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    world_to_clothfunnels_camera_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    clothfunnels_camera_to_view_transform: np.ndarray = dataclasses.field(default=np.eye(3))
    clothfunnels_camera_intrinsics: o3d.camera.PinholeCameraIntrinsic = dataclasses.field(default=None)

    def __post_init__(self):
        if self.option is not None:
            with open(osp.join(self.option.compat.calibration_path, 'world_to_camera_transform.json'), 'r') as f:
                self.world_to_camera_transform = np.array(json.load(f))
                self.camera_to_world_transform = np.linalg.inv(self.world_to_camera_transform)
            with open(osp.join(self.option.compat.calibration_path, 'world_to_left_robot_transform.json'), 'r') as f:
                self.world_to_left_robot_transform = np.array(json.load(f))
            with open(osp.join(self.option.compat.calibration_path, 'world_to_right_robot_transform.json'), 'r') as f:
                self.world_to_right_robot_transform = np.array(json.load(f))

            self.left_robot_to_world_transform = np.linalg.inv(self.world_to_left_robot_transform)
            self.right_robot_to_world_transform = np.linalg.inv(self.world_to_right_robot_transform)

            self.left_robot_base_pos = (self.left_robot_to_world_transform @ np.array([[0., 0., 0., 1.]]).T)[:3, 0]
            self.right_robot_base_pos = (self.right_robot_to_world_transform @ np.array([[0., 0., 0., 1.]]).T)[:3, 0]
            self.virtual_to_world_transform = np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.], [0., 0., 1., 0.03], [0., 0., 0., 1.]])
            self.world_to_virtual_transform = np.linalg.inv(self.virtual_to_world_transform)

            self.virtual_to_camera_transform = self.world_to_camera_transform @ self.virtual_to_world_transform
            self.camera_to_virtual_transform = np.linalg.inv(self.virtual_to_camera_transform)

            self.virtual_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
                osp.join(self.option.compat.calibration_path, 'camera_intrinsic_scanner.json'))

            # read parameters for ClothFunnels camera (square image)
            self.clothfunnels_camera_intrinsics = o3d.io.read_pinhole_camera_intrinsic(
                osp.join(self.option.compat.calibration_path, 'cloth_funnels_camera_intrinsic_720.json'))
            with open(osp.join(self.option.compat.calibration_path, 'cloth_funnels_virtual_to_camera_transform.json'), 'r') as f:
                self.virtual_to_clothfunnels_camera_transform = np.array(json.load(f))
            with open(osp.join(self.option.compat.calibration_path, 'cloth_funnels_camera_to_view_transform.json'), 'r') as f:
                self.clothfunnels_camera_to_view_transform = np.array(json.load(f))
            self.world_to_clothfunnels_camera_transform = self.virtual_to_clothfunnels_camera_transform @ \
                                                          self.world_to_virtual_transform
        else:
            pass

    def virtual_pose_to_world_pose(self, pose_3plus1: np.ndarray) -> RigidTransform:
        xyz = pose_3plus1[np.newaxis, :3]
        virtual_pcd = o3d.geometry.PointCloud()
        virtual_pcd.points = o3d.utility.Vector3dVector(xyz)
        world_pcd = virtual_pcd.transform(self.virtual_to_world_transform)
        xyz_world = np.asarray(world_pcd.points)[0]
        theta = pose_3plus1[-1]
        # TODO: find the correct rotation angle
        r, p, y = 0., 0., theta
        res = RigidTransform(translation=xyz_world, rotation=tr.euler_matrix(r, p, y)[:3, :3], from_frame='robot_tcp')
        return res

    def world_pose_to_virtual_pose(self, pose_world: RigidTransform) -> RigidTransform:
        """
        pose_world: RigidTransform
        """
        rotation, translation = RigidTransform.rotation_and_translation_from_matrix(self.world_to_virtual_transform)
        world_to_virtual_transform = RigidTransform(rotation=rotation,
                                                    translation=translation,
                                                    from_frame='world',
                                                    to_frame='virtual')
        # res = pose_world * world_to_virtual_transform
        res = world_to_virtual_transform * pose_world
        return res


def convert_dict(config: dict):
    new_dict = dict()
    for key, edict_item in config.items():
        if isinstance(edict_item, dict):
            new_dict[key] = convert_dict(config[key])
        else:
            if isinstance(edict_item, np.ndarray):
                new_dict[key] = edict_item.tolist()
            else:
                new_dict[key] = edict_item
    return new_dict


def config_completion(config: Union[Dict, EasyDict, DictConfig, str]) -> Union[DictConfig, ListConfig]:
    if isinstance(config, str):
        option = OmegaConf.load(config)
    elif isinstance(config, dict):
        config = convert_dict(config)
        option = OmegaConf.create(config)
    elif isinstance(config, EasyDict):
        config = convert_dict(dict(config))
        option = OmegaConf.create(config)
    elif isinstance(config, DictConfig):
        option = OmegaConf.create(config)
    else:
        raise NotImplementedError

    # TODO: compatibility with old config
    with open_dict(option):
        # automatically override robot positions by reading calibration files
        option.compat.garment_type = GarmentTypeDef.from_string(option.compat.garment_type)
        with open(osp.join(option.compat.calibration_path, 'world_to_left_robot_transform.json'), 'r') as f:
            left_robot_to_world_transform = np.linalg.inv(np.array(json.load(f)))
        with open(osp.join(option.compat.calibration_path, 'world_to_right_robot_transform.json'), 'r') as f:
            right_robot_to_world_transform = np.linalg.inv(np.array(json.load(f)))

        left_rpy_in_world = Rotation.from_matrix(left_robot_to_world_transform[:3, :3]).as_euler('xyz')
        right_rpy_in_world = Rotation.from_matrix(right_robot_to_world_transform[:3, :3]).as_euler('xyz')

        # print("====================[ DEBUG ]====================")
        # print(option.planning.robot_init_positions, [tuple(left_robot_to_world_transform[:3,3].tolist()), tuple(right_robot_to_world_transform[:3,3].tolist())])
        # print(option.planning.robot_init_orientations, [tuple(left_rpy_in_world),tuple(right_rpy_in_world)])
        # print("====================[  END  ]====================")
        option.planning.robot_init_positions = [tuple(left_robot_to_world_transform[:3, 3].tolist()), tuple(right_robot_to_world_transform[:3, 3].tolist())]
        option.planning.robot_init_orientations = [tuple(left_rpy_in_world.tolist()), tuple(right_rpy_in_world.tolist())]

    return option
