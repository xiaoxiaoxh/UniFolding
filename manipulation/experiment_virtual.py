import sys
import os
import os.path as osp
# sys.path.append(osp.join('..', os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
import numpy as np
from typing import Tuple, Dict, Any, Union, List, Optional
import open3d as o3d
from easydict import EasyDict
from omegaconf import OmegaConf, DictConfig
from learning.components.cloth_dynamics_env import ClothDynamicsEnv
from learning.components.reward import deformable_distance
import json
import dataclasses
from autolab_core import RigidTransform
from autolab_core import transformations as tr
from common.space_util import transform_point_cloud
from common.experiment_base import ExperimentBase
from common.datamodels import ActionMessage, ActionTypeDef, GarmentTypeDef, ObservationMessage, ExceptionMessage, ExecutionErrorTypeDef
from common.metric_utils import get_covered_area_from_particles, get_iou_from_2d_points
import cv2

def config_completion(config: Union[Dict, EasyDict, DictConfig, str]) -> OmegaConf:
    if isinstance(config, str):
        option = OmegaConf.load(config)
    elif isinstance(config, dict):
        option = OmegaConf.create(config)
    elif isinstance(config, DictConfig):
        option = OmegaConf.create(config)
    elif isinstance(config, EasyDict):
        option = OmegaConf.create(config)

    option.compat.garment_type = GarmentTypeDef.from_string(option.compat.garment_type)

    return option

@dataclasses.dataclass
class ExperimentVirtualTransforms:
    option: OmegaConf = dataclasses.field(default=None)
    world_to_camera_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    camera_to_world_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    world_to_left_robot_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    world_to_right_robot_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    left_robot_to_world_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    right_robot_to_world_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    left_robot_base_pos: np.ndarray = dataclasses.field(default=np.array([0.0, 0.0, 0.0]))
    right_robot_base_pos: np.ndarray = dataclasses.field(default=np.array([0.0, 0.0, 0.0]))
    virtual_to_world_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    world_to_unity_transform: np.ndarray = dataclasses.field(default=np.eye(4))
    open3d_to_virtual_transform: np.ndarray = dataclasses.field(default=np.eye(4))

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

            self.virtual_to_world_transform = np.array([[0., 1., 0., 0.],
                                                        [-1., 0., 0., 0.],
                                                        [0., 0., 1., 0.],
                                                        [0., 0., 0., 1.]])
            self.world_to_virtual_transform = np.linalg.inv(self.virtual_to_world_transform)
            self.world_to_unity_transform = np.array([[0., -1., 0., 0.],
                                                      [0., 0., 1., 0.],
                                                      [1., 0., 0., 0.],
                                                      [0., 0., 0., 1.]])
            self.open3d_to_virtual_transform = np.array([[1., 0., 0., 0.],
                                                         [0., 0., -1., 0.],
                                                         [0., 1., 0., 0.],
                                                         [0., 0., 0., 1.]])
            self.unity_camera_local_to_world_transform = self.world_to_unity_transform @ self.camera_to_world_transform
            self.virtual_intrinsics = o3d.io.read_pinhole_camera_intrinsic(
                self.option.compat.camera.virtual_camera_intrinsics_path)
        else:
            pass

    def virtual_pose_to_world_pose(self, pose_3plus1: np.ndarray) -> RigidTransform:
        """
        pose_3plus1: (x, y, z, theta)
        """
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

class ExperimentVirtual(ExperimentBase):
    option: OmegaConf
    env: Optional[ClothDynamicsEnv]
    transforms: ExperimentVirtualTransforms
    def __init__(self, config: Union[Dict, DictConfig, EasyDict, str]):
        self.option: OmegaConf = config_completion(config)

        assert self._init_env() is None

    def _init_env(self) -> Optional[Exception]:
        self.transforms = ExperimentVirtualTransforms(option=self.option)
        assert self.option.environment.env_type == 'ClothDynamics', 'Only ClothDynamicsEnv is supported'
        if self.option.environment.env_type == 'ClothDynamics':
            self.env = ClothDynamicsEnv(camera_intrinsic=self.transforms.virtual_intrinsics,
                                        camera_local_to_world_matrix_unity=self.transforms.unity_camera_local_to_world_transform,
                                        **self.option.environment)
        else:
            raise NotImplementedError
        return None

    def set_category_meta(self, meta_dict: dict):
        self.env.set_category_meta(meta_dict)

    def __del__(self):
        self.env.close()

    def assign_to_arm(self, pose1: RigidTransform, pose2: RigidTransform) -> Tuple[RigidTransform, RigidTransform]:
        """returns tuple with (left arm, right arm)"""

        pose_only1 = pose1[0] if isinstance(pose1, tuple) else pose1
        pose_only2 = pose2[0] if isinstance(pose2, tuple) else pose2

        trans1 = pose_only1.translation if pose_only1 is not None else np.zeros(3)
        trans2 = pose_only2.translation if pose_only2 is not None else np.zeros(3)

        if trans1[1] > trans2[1]:
            self.assign_tcp_frame(pose1, pose2)
            return pose1, pose2
        else:
            self.assign_tcp_frame(pose2, pose1)
            return pose2, pose1

    def is_pose_safe(self, pose1: RigidTransform, pose2: RigidTransform) -> bool:
        return np.linalg.norm(pose1.translation - pose2.translation) > self.option.compat.machine.safe_distance_m

    def is_pose_within_workspace(self, pose: RigidTransform) -> bool:
        # TODO: use config to set workspace
        return True

    def is_pose_reachable(self, pose: RigidTransform, is_left_robot: bool = True) -> bool:
        # TODO: implement this
        return True

    def is_pose_reachable_by_dual_arm(self, pose: RigidTransform) -> Tuple[bool, bool]:
        return self.is_pose_within_workspace(pose) and self.is_pose_reachable(pose, is_left_robot=True), \
            self.is_pose_within_workspace(pose) and self.is_pose_reachable(pose, is_left_robot=False)

    def is_action_executable(self, action_type: ActionTypeDef,
                             poses: Tuple[RigidTransform, RigidTransform, RigidTransform, RigidTransform],
                             verbose=False,
                             return_detailed_err: bool = False) -> Tuple[
        Optional[Dict[str, RigidTransform]], Optional[ExceptionMessage]]:
        """
        Jude whether the input action with 6D poses is executable,
        return Dict, None with transforms (3D poses) if action is valid,
        return None, ExceptionMessage if action is not executable
        """
        pick1, pick2, place1, place2 = poses
        pick1, pick2 = self.assign_to_arm(pick1, pick2)
        place1, place2 = self.assign_to_arm(place1, place2)

        pick1_reachable = self.is_pose_within_workspace(pick1) and self.is_pose_reachable(pick1, is_left_robot=True)
        pick2_reachable = self.is_pose_within_workspace(pick2) and self.is_pose_reachable(pick2, is_left_robot=False)
        if not pick1_reachable and pick2_reachable:
            if verbose:
                logger.debug(f"Skip left pick pose as it's outside workspace: {pick1.translation}")
            return None, ExceptionMessage("Left pick pose is outside workspace",
                                          code=ExecutionErrorTypeDef.UNREACHABLE_PICK_LEFT)
        elif pick1_reachable and not pick2_reachable:
            if verbose:
                logger.debug(f"Skip right pick pose as it's outside workspace: {pick2.translation}")
            return None, ExceptionMessage("Right pick pose is outside workspace",
                                          code=ExecutionErrorTypeDef.UNREACHABLE_PICK_RIGHT)
        elif not pick1_reachable and not pick2_reachable:
            if verbose:
                logger.debug(f"Skip dual-arm pick pose as they are outside workspace: {pick1.translation}, {pick2.translation}")
            return None, ExceptionMessage("Dual-arm pick pose is outside workspace",
                                          code=ExecutionErrorTypeDef.UNREACHABLE_PICK_DUAL)

        if not self.is_pose_safe(pick1, pick2):
            if verbose:
                logger.debug(f"Skip pick poses as they are too close: {pick1.translation}, {pick2.translation}")
            return None, ExceptionMessage("Pick poses are too close", code=ExecutionErrorTypeDef.UNSAFE_FOR_DUAL_ARM)

        # removed the pose adjustment because it makes no sense

        if action_type == ActionTypeDef.FLING:
            pick_left, pick_right = self.assign_to_arm(pick1, pick2)
            # TODO: optimize angles for better reachability
            return {'pick_left': pick_left, 'pick_right': pick_right}, None
        elif action_type in {ActionTypeDef.FOLD_1, ActionTypeDef.FOLD_2, ActionTypeDef.PICK_AND_PLACE, ActionTypeDef.DRAG}:
            self.assign_tcp_frame(pick1, pick2)
            pick_left, pick_right = pick1, pick2
            # TODO: optimize angles for better reachability
            place1_reachable = self.is_pose_within_workspace(place1) and self.is_pose_reachable(place1,
                                                                                                is_left_robot=True)
            place2_reachable = self.is_pose_within_workspace(place2) and self.is_pose_reachable(place2,
                                                                                                is_left_robot=False)
            if not place1_reachable and place2_reachable:
                if verbose:
                    logger.debug(f"Skip left place pose as it's outside workspace: {place1.translation}")
                return None, ExceptionMessage("Left place pose is outside workspace",
                                              code=ExecutionErrorTypeDef.UNREACHABLE_PLACE_LEFT)
            elif place1_reachable and not place2_reachable:
                if verbose:
                    logger.debug(f"Skip right place pose as it's outside workspace: {place2.translation}")
                return None, ExceptionMessage("Right place pose is outside workspace",
                                              code=ExecutionErrorTypeDef.UNREACHABLE_PLACE_RIGHT)
            elif not place1_reachable and not place2_reachable:
                if verbose:
                    logger.debug(
                        f"Skip dual-arm place pose as they are outside workspace: {place1.translation}, {place2.translation}")
                return None, ExceptionMessage("Dual-arm place pose is outside workspace",
                                              code=ExecutionErrorTypeDef.UNREACHABLE_PLACE_DUAL)

            # TODO: more flexible for fold1
            # ignore fold1 place-points safety check
            if not self.is_pose_safe(place1, place2) and action_type != ActionTypeDef.FOLD_1:
                if verbose:
                    logger.debug(f"Skip place poses as they are too close: {place1.translation}, {place2.translation}")
                return None, ExceptionMessage("place poses are too close",
                                              code=ExecutionErrorTypeDef.UNSAFE_FOR_DUAL_ARM)

            self.assign_tcp_frame(place1, place2)
            place_left, place_right = place1, place2
            return {'pick_left': pick_left, 'pick_right': pick_right, 'place_left': place_left, 'place_right': place_right}, None
        else:
            return None, ExceptionMessage(f"Unknown action type: {action_type}", code=None)

    def load_obj(self, obj_path):
        self.env.load_obj(obj_path)

    def random_grasp_and_drop(self):
        self.env.random_grab()

    def random_fold(self):
        self.env.random_fold()

    def fling(self, pick1: RigidTransform, pick2: RigidTransform, **kwargs) -> Optional[ExceptionMessage]:
        # convert from world coordinate system to virtual coordinate system
        pick1_pos_unity = transform_point_cloud(pick1.translation[np.newaxis, :], self.transforms.world_to_unity_transform)[0].astype(np.float32)
        pick2_pos_unity = transform_point_cloud(pick2.translation[np.newaxis, :], self.transforms.world_to_unity_transform)[0].astype(np.float32)
        # TODO: more flexible
        pick1_pos_unity[1] += 0.05  # y-axis offset
        pick2_pos_unity[1] += 0.05  # y-axis offset
        self.env.fling(pick1_pos_unity, pick2_pos_unity)
        return None

    def capture_pcd(self) -> Tuple[ObservationMessage, Optional[Exception]]:
        """
        Capture point cloud from camera, transform the observation into virtual coordinate system
        """
        obs_dict = self.env.get_observation()  # Open3D coordinate system
        mask_img, pc_xyz, pc_nocs, particle_xyz = \
            obs_dict['mask_img'], obs_dict['pc_xyz'], obs_dict['pc_nocs'], obs_dict['particle_xyz']
        pc_xyz = transform_point_cloud(pc_xyz, self.transforms.open3d_to_virtual_transform).astype(np.float32)
        particle_xyz = transform_point_cloud(particle_xyz, self.transforms.open3d_to_virtual_transform).astype(np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        if self.option.compat.camera.vis:
            coorninate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=(0., 0., 0.))
            o3d.visualization.draw_geometries([pcd, coorninate])
        pcd_message = ObservationMessage(mask_img=mask_img,
                                         valid_virtual_pts=pc_xyz,
                                         valid_virtual_pcd=pcd,
                                         valid_nocs_pts=pc_nocs,
                                         particle_xyz=particle_xyz)
        return pcd_message, None

    def execute_action(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        if action.action_type == ActionTypeDef.FLING:
            pick_left, pick_right = action.pick_points[0], action.pick_points[1]
            return self.fling(pick_left, pick_right)
        else:
            raise NotImplementedError

    def get_best_fling_action(self):
        best_grasp_points_open3d = self.env.best_grasp_points  # (2, 3) np.ndarray
        best_grasp_points_virtual = transform_point_cloud(best_grasp_points_open3d,
                                                          self.transforms.open3d_to_virtual_transform)
        best_grasp_points_world = transform_point_cloud(best_grasp_points_virtual,
                                                        self.transforms.virtual_to_world_transform)
        pick1 = RigidTransform(translation=best_grasp_points_world[0])
        pick2 = RigidTransform(translation=best_grasp_points_world[1])
        pick1, pick2 = self.assign_to_arm(pick1, pick2)
        action = ActionMessage(action_type=ActionTypeDef.FLING,
                                 pick_points=[pick1, pick2])
        return action


    def get_deformable_reward(self, obs_message: ObservationMessage) -> dict:
        """factorized reward from Cloth Funnels"""
        # transform the init particles into virtual coordinate system
        init_particles = transform_point_cloud(self.env.init_particle_pos,
                                               self.transforms.open3d_to_virtual_transform)
        max_coverage = get_covered_area_from_particles(init_particles, self.env.particle_radius)

        curr_particles = obs_message.particle_xyz
        curr_mask = obs_message.mask_img

        if self.option.environment.debug:
            coorninate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=(0., 0., 0.))

            init_particle_pcd = o3d.geometry.PointCloud()
            init_particle_pcd.points = o3d.utility.Vector3dVector(init_particles)
            init_particle_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # blue

            curr_particle_pcd = o3d.geometry.PointCloud()
            curr_particle_pcd.points = o3d.utility.Vector3dVector(curr_particles)
            curr_particle_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red
            o3d.visualization.draw_geometries([init_particle_pcd, curr_particle_pcd, coorninate])

        weighted_distance, l2_distance, icp_distance, real_l2_distance, _ = deformable_distance(
            init_particles, curr_particles, max_coverage, self.option.reward.deformable_weight)

        iou = get_iou_from_2d_points(curr_particles[:, :2], init_particles[:, :2],
                                     vis=self.option.environment.debug)
        cur_coverage = get_covered_area_from_particles(curr_particles, self.env.particle_radius)
        normalized_coverage = float(cur_coverage / max_coverage)

        deformable_dict = {
            "weighted_distance": weighted_distance,
            "l2_distance": l2_distance,
            "icp_distance": icp_distance,
            "pointwise_distance": real_l2_distance,
            "iou": iou,
            "normalized_coverage": normalized_coverage
        }
        logger.debug(f"Deformable Reward: {deformable_dict}")
        return deformable_dict

    def get_pick_points_in_virtual(self, action: ActionMessage) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pick points from world coordinate system to virtual coordinate system
        Return a list (left, right) of 6d vector: (X, Y, Z, R, P, Y)
        """
        pick_left, pick_right = action.pick_points[0], action.pick_points[1]
        assert pick_left is not None and pick_right is not None
        pick_left_in_virtual = self.transforms.world_pose_to_virtual_pose(pick_left)
        pick_right_in_virtual = self.transforms.world_pose_to_virtual_pose(pick_right)
        pick_left_numpy = np.concatenate([pick_left_in_virtual.translation, pick_left_in_virtual.euler_angles], axis=-1)
        pick_right_numpy = np.concatenate([pick_right_in_virtual.translation, pick_right_in_virtual.euler_angles], axis=-1)
        return pick_left_numpy, pick_right_numpy



