import sys
import os
import os.path as osp


sys.path.append(osp.join(osp.dirname(__file__),".."))

from planning.controller import DualFliexivController
from planning.utils.pose import pose_to_4x4, pose_to_7D, rotation_about_xyz, pos_rot_to_pose_4x4

from enum import Enum
from loguru import logger
import numpy as np
from typing import Tuple, Dict, Any, Union, List, Optional
from easydict import EasyDict
from omegaconf import OmegaConf, DictConfig, open_dict, ListConfig
from third_party.phoxi import PhoXiCamera
from third_party.filecamera import FileCamera
from third_party.grounded_sam.grounded_sam import GroundedSAM
import open3d as o3d
from autolab_core import RigidTransform
from common.experiment_base import ExperimentBase, ExperimentRealTransforms, config_completion, convert_dict
from common.datamodels import ActionMessage, ActionTypeDef, GarmentTypeDef, ObservationMessage, ExceptionMessage, ExecutionErrorTypeDef
from common.space_util import transform_point_cloud
import math
import hydra

class ExperimentReal(ExperimentBase):
    option: DictConfig
    camera: Union[PhoXiCamera, FileCamera]
    transforms: ExperimentRealTransforms

    def __init__(self, config: Union[Dict, EasyDict, str]):
      
        self.option: DictConfig = config_completion(config)

        """for test api
        """
        assert self._init_camera() is None
        if self.option.compat.use_real_robots:
            assert self._init_robot(config.planning) is None

    def _init_camera(self) -> Optional[Exception]:
        transforms = ExperimentRealTransforms(option=self.option)
        if self.option.compat.camera.use_file_camera:
            self.camera = FileCamera(**self.option.compat.camera, transforms=transforms)
        else:
            self.camera = PhoXiCamera(**self.option.compat.camera, transforms=transforms)
                
        self.camera.start()
        self.transforms = transforms

        self.segmentation_model = GroundedSAM(**self.option.compat.segmentation)
        return None
    
    def __del__(self):
        pass

    def _init_robot(self, config: EasyDict) -> Optional[Exception]:
        print(config)
        self.controller = DualFliexivController(config=config)
        # make sure the robot type(left or right)
        self.pick_type = None
        return None

    """for test api
    """
    # def __del__(self):
    #     self.camera.stop()

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
        robot_distance = np.linalg.norm(pose1.translation[:2] - pose2.translation[:2])
        if robot_distance < self.option.compat.machine.safe_distance_m:
            return False
        robot_y_distance = np.abs(pose1.translation[1]-pose2.translation[1])
        if robot_y_distance < self.option.compat.machine.safe_y_distance_cm:
            return False
        return True

    def is_pose_reachable(self, pose: RigidTransform, is_left_robot: bool = True) -> bool:
        if is_left_robot == True:
            robot_base_to_robot_distance = math.sqrt((pose.translation[0:2]-self.transforms.left_robot_base_pos[0:2])@
                                                     (pose.translation[0:2]-self.transforms.left_robot_base_pos[0:2]).T)
            if pose.translation[0] < self.option.compat.machine.left_workspace_min_x or \
                robot_base_to_robot_distance > self.option.compat.machine.left_workspace[1]:
                return False
            if pose.translation[1] < self.option.compat.machine.left_workspace_y_limits[0] or \
                    pose.translation[1] > self.option.compat.machine.left_workspace_y_limits[1]:
                return False
            return True
        else:
            robot_base_to_robot_distance = math.sqrt((pose.translation[0:2]-self.transforms.right_robot_base_pos[0:2])@
                                                     (pose.translation[0:2]-self.transforms.right_robot_base_pos[0:2]).T)
            if pose.translation[0] < self.option.compat.machine.right_workspace_min_x or \
               robot_base_to_robot_distance > self.option.compat.machine.right_workspace[1]:
                return False
            if pose.translation[1] < self.option.compat.machine.right_workspace_y_limits[0] or \
                    pose.translation[1] > self.option.compat.machine.right_workspace_y_limits[1]:
                return False
            return True

    def is_pose_draggable(self, pose: RigidTransform, is_left_robot: bool = True) -> bool:
        if is_left_robot == True:
            robot_base_to_robot_distance = math.sqrt((pose.translation[0:2]-self.transforms.left_robot_base_pos[0:2])@
                                                     (pose.translation[0:2]-self.transforms.left_robot_base_pos[0:2]).T)
            logger.debug(f"left drag pose {pose.translation[0:2]}, robot base distance {robot_base_to_robot_distance}")
            return robot_base_to_robot_distance > self.option.compat.machine.normal_drag.min_robot_distance_for_drag and \
                self.option.compat.machine.normal_drag.y_limits_before_drag[0] < pose.translation[1]
        else:
            robot_base_to_robot_distance = math.sqrt((pose.translation[0:2]-self.transforms.right_robot_base_pos[0:2])@
                                                     (pose.translation[0:2]-self.transforms.right_robot_base_pos[0:2]).T)
            logger.debug(f"right drag pose {pose.translation[0:2]}, robot base distance {robot_base_to_robot_distance}")
            return robot_base_to_robot_distance > self.option.compat.machine.normal_drag.min_robot_distance_for_drag and \
                pose.translation[1] < self.option.compat.machine.normal_drag.y_limits_before_drag[1]

    def is_pose_in_center(self, pose: RigidTransform, is_left_robot: bool = True) -> bool:
        if is_left_robot == True:
            return self.option.compat.machine.normal_drag.y_limits_before_drag[0] < pose.translation[1]
        else:
            return pose.translation[1] < self.option.compat.machine.normal_drag.y_limits_before_drag[1]

    def is_pose_too_far(self, pose: RigidTransform, is_left_robot: bool = True) -> bool:
        if is_left_robot:
            robot_base_to_robot_distance = math.sqrt((pose.translation[0:2]-self.transforms.left_robot_base_pos[0:2])@
                                                     (pose.translation[0:2]-self.transforms.left_robot_base_pos[0:2]).T)
            return robot_base_to_robot_distance > self.option.compat.machine.left_workspace[1]
        else:
            robot_base_to_robot_distance = math.sqrt((pose.translation[0:2]-self.transforms.right_robot_base_pos[0:2])@
                                                     (pose.translation[0:2]-self.transforms.right_robot_base_pos[0:2]).T)
            return robot_base_to_robot_distance > self.option.compat.machine.right_workspace[1]

    def is_pose_too_near(self, pose: RigidTransform, is_left_robot: bool = True) -> bool:
        if is_left_robot:
            robot_base_to_robot_distance = math.sqrt((pose.translation[0:2]-self.transforms.left_robot_base_pos[0:2])@
                                                     (pose.translation[0:2]-self.transforms.left_robot_base_pos[0:2]).T)
            return robot_base_to_robot_distance < self.option.compat.machine.left_workspace[0]
        else:
            robot_base_to_robot_distance = math.sqrt((pose.translation[0:2]-self.transforms.right_robot_base_pos[0:2])@
                                                     (pose.translation[0:2]-self.transforms.right_robot_base_pos[0:2]).T)
            return robot_base_to_robot_distance < self.option.compat.machine.right_workspace[0]

    def is_pose_within_workspace(self, pose: RigidTransform) -> bool:
        _x = self.option.compat.machine.x_lim_m[0] < pose.translation[0] < self.option.compat.machine.x_lim_m[1]
        _y = self.option.compat.machine.y_lim_m[0] < pose.translation[1] < self.option.compat.machine.y_lim_m[1]
        _z = self.option.compat.machine.z_lim_m[0] < pose.translation[2] < self.option.compat.machine.z_lim_m[1]
        return _x and _y and _z
    
    def virtual_pose_to_world_pose(self, pose_in_world_space_6d: np.ndarray) -> RigidTransform:
        return self.transforms.virtual_pose_to_world_pose(pose_in_world_space_6d)

    def is_pose_reachable_by_dual_arm(self, pose: RigidTransform) -> Tuple[bool, bool]:
        return self.is_pose_within_workspace(pose) and self.is_pose_reachable(pose, is_left_robot=True), \
            self.is_pose_within_workspace(pose) and self.is_pose_reachable(pose, is_left_robot=False)

    def is_action_executable(self, action_type: ActionTypeDef,
                             poses: Tuple[RigidTransform, RigidTransform, RigidTransform, RigidTransform],
                             verbose=True,
                             return_detailed_err: bool = False) -> Tuple[Optional[Dict[str, RigidTransform]], Optional[ExceptionMessage]]:
        """
        Jude whether the input action with 6D poses is executable,
        return Dict with transforms (3D poses)
        return None, ExceptionMessage if action is not executable
        """
        pick1, pick2, place1, place2 = poses
        pick1, pick2 = self.assign_to_arm(pick1, pick2)
        self.assign_tcp_frame(place1, place2)  # we do not need to automatically assign arms for place points

        # temorary dict for error handlers
        err_transform_dict = {'pick_left': pick1, 'pick_right': pick2, 'place_left': place1, 'place_right': place2}

        if return_detailed_err:
            pick1_too_far = self.is_pose_too_far(pick1, is_left_robot=True)        
            pick2_too_far = self.is_pose_too_far(pick2, is_left_robot=False)
            if pick1_too_far and not pick2_too_far:
                logger.debug(f"left pick pose is too far from robot base : {pick1.translation}")
                return err_transform_dict, ExceptionMessage(f"pick_left is too far from robot base", 
                                                            code=ExecutionErrorTypeDef.TOO_FAR_FOR_PICK_LEFT)
            elif not pick1_too_far and pick2_too_far:
                logger.debug(f"right pick  pose is too far from robot base : {pick2.translation}")
                return err_transform_dict, ExceptionMessage(f"pick_right is too far from robot base", 
                                                            code=ExecutionErrorTypeDef.TOO_FAR_FOR_PICK_RIGHT)
            elif pick1_too_far and pick2_too_far:
                logger.debug(f"both pick  pose are too far from robot base : {pick1.translation}, {pick2.translation}")
                return err_transform_dict, ExceptionMessage(f"both pick  pose are too far from robot base", 
                                                            code=ExecutionErrorTypeDef.TOO_FAR_FOR_PICK_DUAL)
            # TODO: implement too-near error
        pick1_reachable = self.is_pose_within_workspace(pick1) and self.is_pose_reachable(pick1, is_left_robot=True)        
        pick2_reachable = self.is_pose_within_workspace(pick2) and self.is_pose_reachable(pick2, is_left_robot=False)        
        if not pick1_reachable and pick2_reachable:
            logger.debug(f"Skip left pick pose as it's outside workspace: {pick1.translation}")
            return err_transform_dict, ExceptionMessage("Left pick pose is outside workspace",
                                          code=ExecutionErrorTypeDef.UNREACHABLE_PICK_LEFT)
        elif pick1_reachable and not pick2_reachable:
            logger.debug(f"Skip right pick pose as it's outside workspace: {pick2.translation}")
            return err_transform_dict, ExceptionMessage("Right pick pose is outside workspace",
                                          code=ExecutionErrorTypeDef.UNREACHABLE_PICK_RIGHT)
        elif not pick1_reachable and not pick2_reachable:
            logger.debug(f"Skip dual-arm pick pose as they are outside workspace: {pick1.translation}, {pick2.translation}")
            return err_transform_dict, ExceptionMessage("Dual-arm pick pose is outside workspace",
                                          code=ExecutionErrorTypeDef.UNREACHABLE_PICK_DUAL)

        if not self.is_pose_safe(pick1, pick2):
            logger.debug(f"Skip pick poses as they are too close: {pick1.translation}, {pick2.translation}")
            return err_transform_dict, ExceptionMessage("Pick poses are too close", code=ExecutionErrorTypeDef.UNSAFE_FOR_DUAL_ARM)

        # removed the pose adjustment because it makes no sense

        if action_type == ActionTypeDef.FLING:
            self.assign_tcp_frame(pick1, pick2)
            pick_left, pick_right = self.assign_to_arm(pick1, pick2)
            # TODO: optimize angles for better reachability
            return {'pick_left': pick_left, 'pick_right': pick_right}, None
        elif action_type in {ActionTypeDef.FOLD_1, ActionTypeDef.FOLD_2, ActionTypeDef.PICK_AND_PLACE,
                             ActionTypeDef.PICK_AND_PLACE_SINGLE, ActionTypeDef.DRAG}:
            # TODO: handle PICK_AND_PLACE_SINGLE in a better way
            self.assign_tcp_frame(pick1, pick2)
            pick_left, pick_right = pick1, pick2

            if return_detailed_err:
                place1_too_far = self.is_pose_too_far(place1, is_left_robot=True)        
                place2_too_far = self.is_pose_too_far(place2, is_left_robot=False)
                if place1_too_far and not place2_too_far:
                    logger.debug(f"left place pose is too far from robot base : {place1.translation}")
                    return err_transform_dict, ExceptionMessage(f"place_left is too far from robot base", 
                                                                code=ExecutionErrorTypeDef.TOO_FAR_FOR_PLACE_LEFT)
                elif place2_too_far and not place1_too_far:
                    logger.debug(f"right place  pose is too far from robot base : {place2.translation}")
                    return err_transform_dict, ExceptionMessage(f"place_right is too far from robot base",
                                                                code=ExecutionErrorTypeDef.TOO_FAR_FOR_PLACE_RIGHT)
                elif place1_too_far and place2_too_far:
                    logger.debug(f"both place  pose are too far from robot base : {place1.translation}, {place2.translation}")
                    return err_transform_dict, ExceptionMessage(f"both place  pose are too far from robot base",
                                                                code=ExecutionErrorTypeDef.TOO_FAR_FOR_PLACE_DUAL)

            # TODO: optimize angles for better reachability
            place1_reachable = self.is_pose_within_workspace(place1) and self.is_pose_reachable(place1,
                                                                                                is_left_robot=True)
            place2_reachable = self.is_pose_within_workspace(place2) and self.is_pose_reachable(place2,
                                                                                                is_left_robot=False)
            if return_detailed_err:
                if not place1_reachable and place2_reachable:
                    logger.debug(f"Skip left place pose as it's outside workspace: {place1.translation}")
                    return err_transform_dict, ExceptionMessage("Left place pose is outside workspace",
                                                code=ExecutionErrorTypeDef.UNREACHABLE_PLACE_LEFT)
                elif place1_reachable and not place2_reachable:
                    logger.debug(f"Skip right place pose as it's outside workspace: {place2.translation}")
                    return err_transform_dict, ExceptionMessage("Right place pose is outside workspace",
                                                code=ExecutionErrorTypeDef.UNREACHABLE_PLACE_RIGHT)
                elif not place1_reachable and not place2_reachable:
                    logger.debug(
                        f"Skip dual-arm place pose as they are outside workspace: {place1.translation}, {place2.translation}")
                    return err_transform_dict, ExceptionMessage("Dual-arm place pose is outside workspace",
                                                code=ExecutionErrorTypeDef.UNREACHABLE_PLACE_DUAL)

            # ignore fold1 place-points safety check
            if not self.is_pose_safe(place1, place2) and action_type not in \
                    (ActionTypeDef.FOLD_1, ActionTypeDef.PICK_AND_PLACE_SINGLE):
                logger.debug(f"Skip place poses as they are too close: {place1.translation}, {place2.translation}")
                return err_transform_dict, ExceptionMessage("place poses are too close",
                                              code=ExecutionErrorTypeDef.UNSAFE_FOR_DUAL_ARM)

            self.assign_tcp_frame(place1, place2)
            place_left, place_right = place1, place2
            return {'pick_left': pick_left, 'pick_right': pick_right, 'place_left': place_left, 'place_right': place_right}, None
        else:
            return err_transform_dict, ExceptionMessage(f"Unknown action type: {action_type}", code=None)

    def get_drag_hybrid_poses_from_point_cloud(self, pc_xyz: np.ndarray) \
            -> Tuple[Optional[Dict[str, RigidTransform]], Optional[ExceptionMessage]]:
        """
        Get the drag poses from point cloud for drag_hybrid action with two robot arms in world space

        :param pc_xyz: (N, 3) array of point cloud in virtual coordinate system
        return dict of RigidTransform, None if action is valid,
        return None, ExceptionMessage if action is not executable
        """
        pc_xyz_world = transform_point_cloud(pc_xyz, self.transforms.virtual_to_world_transform)
        min_xyz = pc_xyz_world.min(axis=0)
        max_xyz = pc_xyz_world.max(axis=0)
        min_x, max_x = min_xyz[0], max_xyz[0]
        min_y, max_y = min_xyz[1], max_xyz[1]

        centroid = pc_xyz_world.mean(axis=0)
        
        y_range = max_y - min_y
        left_y = centroid[1] + y_range * self.option.compat.machine.hybrid_drag.y_distance_ratio
        right_y = centroid[1] - y_range * self.option.compat.machine.hybrid_drag.y_distance_ratio

        left_pick_valid_idxs = np.abs(pc_xyz_world[:, 1] - left_y) < self.option.compat.machine.hybrid_drag.y_tolerance
        right_pick_valid_idxs = np.abs(pc_xyz_world[:, 1] - right_y) < self.option.compat.machine.hybrid_drag.y_tolerance
        left_pick_valid_pts = pc_xyz_world[left_pick_valid_idxs, :]
        right_pick_valid_pts = pc_xyz_world[right_pick_valid_idxs, :]
        assert left_pick_valid_pts.shape[0] > 0 and right_pick_valid_pts.shape[0] > 0, \
            f"left_pick_valid_pts.shape: {left_pick_valid_pts.shape}, right_pick_valid_pts.shape: {right_pick_valid_pts.shape}"

        min_valid_xyz = pc_xyz_world[left_pick_valid_idxs, :].min(axis=0)
        max_valid_xyz = pc_xyz_world[left_pick_valid_idxs, :].max(axis=0)
        valid_min_x, valid_max_x = min_valid_xyz[0], max_valid_xyz[0]
        valid_min_y, valid_max_y = min_valid_xyz[1], max_valid_xyz[1]

        left_pick_min_x = left_pick_valid_pts[:, 0].min()
        right_pick_min_x = right_pick_valid_pts[:, 0].min()
        left_pick_max_x = left_pick_valid_pts[:, 0].max()
        right_pick_max_x = right_pick_valid_pts[:, 0].max()
        left_pick_x_range = left_pick_max_x - left_pick_min_x
        right_pick_x_range = right_pick_max_x - right_pick_min_x

        left_pick_x = left_pick_min_x + left_pick_x_range * self.option.compat.machine.hybrid_drag.x_distance_ratio
        right_pick_x = right_pick_min_x + right_pick_x_range * self.option.compat.machine.hybrid_drag.x_distance_ratio

        left_pick_point = np.array([left_pick_x, left_y, centroid[2]])
        right_pick_point = np.array([right_pick_x, right_y, centroid[2]])

        left_drag_distance = valid_min_x - self.option.compat.machine.hybrid_drag.min_x_after_drag
        right_drag_distance = valid_min_x - self.option.compat.machine.hybrid_drag.min_x_after_drag
        
        left_place_point = left_pick_point + np.array([-left_drag_distance, 0., 0.])
        right_place_point = right_pick_point + np.array([-right_drag_distance, 0., 0.])

        pick1, pick2 = RigidTransform(translation=left_pick_point), RigidTransform(translation=right_pick_point)
        place1, place2 = RigidTransform(translation=left_place_point), RigidTransform(translation=right_place_point)
        # assign pick points and place points to left and right arms
        pick_left, pick_right = self.assign_to_arm(pick1, pick2)
        place_left, place_right = self.assign_to_arm(place1, place2)
        self.assign_tcp_frame(pick_left, pick_right)
        self.assign_tcp_frame(place_left, place_right)

        # TODO: handle unreachable errors
        
        return {'pick_left': pick_left, 'pick_right': pick_right, 'place_left': place_left, 'place_right': place_right}, None

    def get_drag_poses_from_target_points(self,
                                          target_pick_points_world: np.ndarray,
                                          pc_xyz_world: np.ndarray) \
            -> Tuple[Optional[Dict[str, RigidTransform]], Optional[ExceptionMessage]]:
        """
        Get the best pick and place poses for drag action with two robot arms in world space.
        Select pick points from the line between the robot base and the target pick points (too far to reach)

        :param target_pick_points_world: (2, 3) array of target pick points in world coordinate system
        :param pc_xyz_world: (N, 3) array of point cloud xyz in world coordinate system

        return dict of RigidTransform, None if action is valid,
        return None, ExceptionMessage if action is not executable
        """
        pick_target_left, pick_target_right = RigidTransform(translation=target_pick_points_world[0]), \
            RigidTransform(translation=target_pick_points_world[1])
        if not self.is_pose_in_center(pick_target_left, is_left_robot=True) or \
                not self.is_pose_in_center(pick_target_right, is_left_robot=False):
            # the target pick points are very bad, give up drag action now
            logger.warning('Cannot find any valid drag positions for the robot! (possibly due to grasp failure of the last experiment)')
            return None, ExceptionMessage('Cannot find any valid poses',
                                          code=ExecutionErrorTypeDef.UNREACHABLE_FOR_DRAG)

        # calculate line vectors
        left_line_vector = pick_target_left.translation - self.transforms.left_robot_base_pos
        right_line_vector = pick_target_right.translation - self.transforms.right_robot_base_pos
        left_line_len = np.linalg.norm(left_line_vector)
        right_line_len = np.linalg.norm(right_line_vector)
        # calculate line directions
        left_line_dir = left_line_vector / left_line_len
        right_line_dir = right_line_vector / right_line_len
        # calculate pick distance
        min_safe_robot_distance = self.option.compat.machine.normal_drag.min_robot_distance_for_drag + 0.01  # for safety
        left_pick_distance = max(self.option.compat.machine.normal_drag.line_ratio * left_line_len,
                                 min_safe_robot_distance)
        right_pick_distance = max(self.option.compat.machine.normal_drag.line_ratio * right_line_len,
                                 min_safe_robot_distance)
        # calculate pick points and place points for drag action
        left_pick_point = self.transforms.left_robot_base_pos + left_pick_distance * left_line_dir
        right_pick_point = self.transforms.right_robot_base_pos + right_pick_distance * right_line_dir
        left_place_point = self.transforms.left_robot_base_pos + min_safe_robot_distance * left_line_dir
        right_place_point = self.transforms.right_robot_base_pos + min_safe_robot_distance * right_line_dir

        # find the closet point on the point cloud for pick points
        # calculate distance to left and right pick point
        left_dist_all = np.linalg.norm(left_pick_point[np.newaxis, :] - pc_xyz_world, axis=1)
        right_dist_all = np.linalg.norm(right_pick_point[np.newaxis, :] - pc_xyz_world, axis=1)
        left_pick_point = pc_xyz_world[np.argmin(left_dist_all), :]
        right_pick_point = pc_xyz_world[np.argmin(right_dist_all), :]

        pick_left, pick_right = RigidTransform(translation=left_pick_point), RigidTransform(
            translation=right_pick_point)
        place_left, place_right = RigidTransform(translation=left_place_point), RigidTransform(
            translation=right_place_point)
        self.assign_tcp_frame(pick_left, pick_right)
        self.assign_tcp_frame(place_left, place_right)

        if not self.is_pose_reachable(pick_left, is_left_robot=True) or \
                not self.is_pose_reachable(pick_right, is_left_robot=False) or \
                not self.is_pose_safe(pick_left, pick_right) or \
                not self.is_pose_draggable(pick_left, is_left_robot=True) or \
                not self.is_pose_draggable(pick_right, is_left_robot=False):
            logger.warning('Cannot find any valid drag positions for the robot!')
            return None, ExceptionMessage('Cannot find any valid poses',
                                          code=ExecutionErrorTypeDef.UNREACHABLE_FOR_DRAG)

        if self.option.compat.debug:
            import matplotlib
            matplotlib.use('TKAgg')
            import matplotlib.pyplot as plt

            left_robot_min_radius = self.option.compat.machine.left_workspace[0]
            right_robot_min_radius = self.option.compat.machine.right_workspace[0]

            left_pick_point_2d = left_pick_point[:2]  # (x, y)
            right_pick_point_2d = right_pick_point[:2]  # (x, y)
            left_place_point_2d = left_place_point[:2]  # (x, y)
            right_place_point_2d = right_place_point[:2]  # (x, y)
            left_robot_base_2d = self.transforms.left_robot_base_pos[:2]  # (x, y)
            right_robot_base_2d = self.transforms.right_robot_base_pos[:2]  # (x, y)

            ax = plt.gca()
            # draw the left robot circle and right robot circle
            left_robot_circle = plt.Circle(left_robot_base_2d, left_robot_min_radius, color='black', fill=False)
            ax.add_patch(left_robot_circle)
            right_robot_circle = plt.Circle(right_robot_base_2d, right_robot_min_radius, color='black', fill=False)
            ax.add_patch(right_robot_circle)
            # draw the pick point and place point
            left_pick_point = plt.Circle(left_pick_point_2d, 0.05, color='red', fill=True)
            ax.add_patch(left_pick_point)
            right_pick_point = plt.Circle(right_pick_point_2d, 0.05, color='red', fill=True)
            ax.add_patch(right_pick_point)
            left_place_point = plt.Circle(left_place_point_2d, 0.05, color='blue', fill=True)
            ax.add_patch(left_place_point)
            right_place_point = plt.Circle(right_place_point_2d, 0.05, color='blue', fill=True)
            ax.add_patch(right_place_point)
            # draw the left robot base and right robot base
            left_robot_base = plt.Circle(left_robot_base_2d, 0.05, color='green', fill=True)
            ax.add_patch(left_robot_base)
            right_robot_base = plt.Circle(right_robot_base_2d, 0.05, color='green', fill=True)
            ax.add_patch(right_robot_base)
            plt.show()

        return {'pick_left': pick_left, 'pick_right': pick_right, 'place_left': place_left,
                'place_right': place_right}, None

    def get_pick_points_in_virtual(self, action: ActionMessage) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pick points from world coordinate system to virtual coordinate system
        Return a list (left, right) of 6d vector: (X, Y, Z, R, P, Y)
        """
        pick_left, pick_right = action.pick_points[0], action.pick_points[1]
        if pick_left is None:
            pick_left_numpy = np.zeros((6, )).astype(np.float32)
        else:
            pick_left_in_virtual = self.transforms.world_pose_to_virtual_pose(pick_left)
            pick_left_numpy = np.concatenate([pick_left_in_virtual.translation, pick_left_in_virtual.euler_angles],
                                             axis=-1)

        if pick_right is None:
            pick_right_numpy = np.zeros((6, )).astype(np.float32)
        else:
            pick_right_in_virtual = self.transforms.world_pose_to_virtual_pose(pick_right)
            pick_right_numpy = np.concatenate([pick_right_in_virtual.translation, pick_right_in_virtual.euler_angles], axis=-1)
        return pick_left_numpy, pick_right_numpy

    def get_place_points_in_virtual(self, action: ActionMessage) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert pick points from world coordinate system to virtual coordinate system
        Return a list (left, right) of 6d vector: (X, Y, Z, R, P, Y)
        """
        place_left, place_right = action.place_points[0], action.place_points[1]
        if place_left is None:
            place_left_numpy = np.zeros((6, )).astype(np.float32)
        else:
            place_left_in_virtual = self.transforms.world_pose_to_virtual_pose(place_left)
            place_left_numpy = np.concatenate([place_left_in_virtual.translation, place_left_in_virtual.euler_angles],
                                              axis=-1)
        if place_right is None:
            place_right_numpy = np.zeros((6, )).astype(np.float32)
        else:
            place_right_in_virtual = self.transforms.world_pose_to_virtual_pose(place_right)
            place_right_numpy = np.concatenate([place_right_in_virtual.translation, place_right_in_virtual.euler_angles], axis=-1)
        return place_left_numpy, place_right_numpy

    def capture_pcd(self) -> Tuple[ObservationMessage, Optional[Exception]]:
        """
        returns raw pcd as 3rd(np.ndarray) and 4th(o3d.geometry.PointCloud) return value
        """
        # get RGB image and Point Cloud from cameras
        rgb_img = self.camera.capture_rgb()  
        camera_pcd = self.camera.capture_pcd()

        pc_xyz = np.asarray(camera_pcd.points).copy()
        # get raw pcd with colors
        pc_rgb = PhoXiCamera.colorize_point_cloud(pc_xyz, rgb_img,
                                                  self.camera.external_intrinsics,
                                                  self.camera.external_extrinsics)
        camera_pcd.colors = o3d.utility.Vector3dVector(pc_rgb / 255.0)
        raw_virtual_pcd = camera_pcd.transform(self.transforms.camera_to_virtual_transform)
        raw_virtual_pts = np.asarray(raw_virtual_pcd.points).astype(np.float32)
        # get masked pcd
        masks = self.segmentation_model.predict(rgb_img)  # (k, h, w)
        mask_sum = masks.sum(axis=-1).sum(axis=-1)  # (k, )
        h, w = masks.shape[1:]
        # filter mask with very large area (probably table)
        mask_sum[mask_sum > h * w * self.option.compat.camera.max_mask_area_ratio] = 0
        max_mask_idx = np.argsort(mask_sum)[::-1][0]
        mask_img = np.transpose(masks[max_mask_idx, :, :][np.newaxis, :, :]
                                .repeat(3, axis=0), (1, 2, 0)).astype(np.uint8)  # (h, w, 3)
        pc_mask_color = PhoXiCamera.colorize_point_cloud(pc_xyz, mask_img,
                                                         self.camera.external_intrinsics, self.camera.external_extrinsics)
        valid_idxs = pc_mask_color[:, 0] > 0
        # generate Open3D PointCloud
        valid_virtual_pcd = o3d.geometry.PointCloud()
        valid_virtual_pcd.points = o3d.utility.Vector3dVector(raw_virtual_pts[valid_idxs, :])
        valid_virtual_pcd.colors = o3d.utility.Vector3dVector(pc_rgb[valid_idxs, :] / 255.0)
        valid_virual_pts = np.asarray(valid_virtual_pcd.points).astype(np.float32)
        # generate reprojected RGB, depth map and mask (for ClothFunnels and other baselines)
        reprojected_rgb, reprojected_depth, reprojected_mask = PhoXiCamera.generate_reprojected_rgb_depth_map(
            raw_virtual_pcd, valid_idxs,
            virtual_intrinsics=self.transforms.clothfunnels_camera_intrinsics,
            virtual_extrinsics=self.transforms.virtual_to_clothfunnels_camera_transform,
            plane_point_uvs=np.asarray(self.option.compat.camera.plane_point_uvs).astype(np.int32),
            vis=self.option.compat.camera.vis)
        res = ObservationMessage(valid_virual_pts, valid_virtual_pcd, raw_virtual_pts, raw_virtual_pcd,
                                 mask_img, rgb_img,
                                 projected_rgb_img=reprojected_rgb,
                                 projected_depth_img=reprojected_depth,
                                 projected_mask_img=reprojected_mask)

        if self.option.compat.camera.vis:
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([coord, raw_virtual_pcd])
            o3d.visualization.draw_geometries([coord, valid_virtual_pcd])

        return res, None

    def is_garment_reachable(self, mask: np.ndarray) -> bool:
        """
        Check whether the garment is reachable according to mask information

        @params: mask (H, W, 3) uint8, np.ndarray
        """
        limits = self.option.compat.machine.image_width_ratio_limits_for_reachability_det
        mask_single = mask[:, :, 0]
        h, w = mask_single.shape[:2]
        valid_idxs = np.nonzero(mask_single)
        # check whether the garment center is in the center region of the table
        return w * limits[0] <= valid_idxs[1].mean() <= w * limits[1]

    def is_garment_on_table(self, mask: np.ndarray) -> bool:
        """
        Check whether the garment is still on the table according to mask information

        @params: mask (H, W, 3) uint8, np.ndarray
        """
        mask_single = mask[:, :, 0]
        mask_sum = mask_single.sum(axis=-1).sum(axis=-1)  # (k, )
        h, w = mask_single.shape
        # if the mask area is too large, it probably covers the whole table,
        # and in this case the garment is probably not on the table
        return mask_sum <= h * w * self.option.compat.camera.max_mask_area_ratio

    def _handle_home(self, _: ActionMessage) -> Optional[ExceptionMessage]:
        self.controller.move_home_with_plan()
        return None
    
    def _handle_done(self, _: ActionMessage) -> Optional[ExceptionMessage]:
        self.controller.move_home_with_plan()

    def _handle_fling(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        if action.action_type != ActionTypeDef.FLING:
            return ExceptionMessage("Action type is wrong", -1)
     
        left_translation,left_quaterion= action.pick_points[0].translation,action.pick_points[0].quaternion
        right_translation,right_quaterion= action.pick_points[1].translation,action.pick_points[1].quaternion

        left_pose = np.concatenate([left_translation,left_quaterion])
        right_pose = np.concatenate([right_translation,right_quaterion])
        
        poses = np.array([left_pose,right_pose])
  
        logger.debug(f"fling action poses:    pick_left:{left_pose}, pick_right:{right_pose}")
        exe_result = self.controller.execute_dual_fling(poses)

        return exe_result

    def _handle_drag_normal(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        pick_left_trans,pick_left_quaterion = action.pick_points[0].translation, action.pick_points[0].quaternion
        pick_right_trans,pick_right_quaterion = action.pick_points[1].translation, action.pick_points[1].quaternion
        place_left_trans,place_left_quaterion = action.place_points[0].translation, action.place_points[0].quaternion 
        place_right_trans,place_right_quaterion = action.place_points[1].translation, action.place_points[1].quaternion
    
        pick_left = np.concatenate([pick_left_trans,pick_left_quaterion]).tolist()
        pick_right = np.concatenate([pick_right_trans,pick_right_quaterion]).tolist()
        place_left = np.concatenate([place_left_trans,place_left_quaterion]).tolist()
        place_right = np.concatenate([place_right_trans,place_right_quaterion]).tolist()
            
        logger.debug(f"drag normal poses:    pick_left:{pick_left}, pick_right:{pick_right}, place_left:{place_left}, place_right:{place_right},")
        exe_result = self.controller.execute_drag_normal(np.array([pick_left,pick_right,place_left,place_right]))
        return exe_result

    def _handle_drag_hybrid(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        pick_left_trans, pick_left_quaterion = action.pick_points[0].translation, action.pick_points[0].quaternion
        pick_right_trans, pick_right_quaterion = action.pick_points[1].translation, action.pick_points[1].quaternion
        place_left_trans, place_left_quaterion = action.place_points[0].translation, action.place_points[0].quaternion
        place_right_trans, place_right_quaterion = action.place_points[1].translation, action.place_points[1].quaternion

        pick_left = np.concatenate([pick_left_trans, pick_left_quaterion]).tolist()
        pick_right = np.concatenate([pick_right_trans, pick_right_quaterion]).tolist()
        place_left = np.concatenate([place_left_trans, place_left_quaterion]).tolist()
        place_right = np.concatenate([place_right_trans, place_right_quaterion]).tolist()

        logger.debug(f"drag hybrid poses:    pick_left:{pick_left}, pick_right:{pick_right}, place_left:{place_left}, place_right:{place_right},")

        trans_l = pick_left_trans - place_left_trans
        trans_l[2] = 0.  # set z-offset as 0


        trans_r = pick_right_trans - place_right_trans
        trans_r[2] = 0.  # set z-offset as 0
   
        exe_result = self.controller.execute_drag_with_hybrid(np.array([pick_left,pick_right,place_left,place_right]),
                                                              trans_l, trans_r)
        return exe_result

    def _handle_fold_1(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        pick_left_trans,pick_left_quaterion = action.pick_points[0].translation, action.pick_points[0].quaternion
        pick_right_trans,pick_right_quaterion = action.pick_points[1].translation, action.pick_points[1].quaternion
        place_left_trans,place_left_quaterion = action.place_points[0].translation, action.place_points[0].quaternion 
        place_right_trans,place_right_quaterion = action.place_points[1].translation, action.place_points[1].quaternion
    
        pick_left = np.concatenate([pick_left_trans,pick_left_quaterion]).tolist()
        pick_right = np.concatenate([pick_right_trans,pick_right_quaterion]).tolist()
        place_left = np.concatenate([place_left_trans,place_left_quaterion]).tolist()
        place_right = np.concatenate([place_right_trans,place_right_quaterion]).tolist()
        
        logger.debug(f"fold one poses:    pick_left:{pick_left}, pick_right:{pick_right}, place_left:{place_left}, place_right:{place_right},")

        exe_result = self.controller.execute_fold_one(np.array([pick_left,pick_right,place_left,place_right]))
        # exe_result = self.controller.execute_calibration_one(np.array([pick_left,pick_right,place_left,place_right]))
        
        return exe_result

    def _handle_fold_2(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        pick_left_trans,pick_left_quaterion = action.pick_points[0].translation, action.pick_points[0].quaternion
        pick_right_trans,pick_right_quaterion = action.pick_points[1].translation, action.pick_points[1].quaternion
        place_left_trans,place_left_quaterion = action.place_points[0].translation, action.place_points[0].quaternion 
        place_right_trans,place_right_quaterion = action.place_points[1].translation, action.place_points[1].quaternion
        pick_left = np.concatenate([pick_left_trans,pick_left_quaterion]).tolist()
        pick_right = np.concatenate([pick_right_trans,pick_right_quaterion]).tolist()
        place_left = np.concatenate([place_left_trans,place_left_quaterion]).tolist()
        place_right = np.concatenate([place_right_trans,place_right_quaterion]).tolist()

        logger.debug(f"fold two poses:    pick_left:{pick_left}, pick_right:{pick_right}, place_left:{place_left}, place_right:{place_right},")
        exe_result = self.controller.execute_fold_two(np.array([pick_left,pick_right,place_left,place_right]))
        return exe_result

    def _handle_pick_and_place(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        pick_left_trans,pick_left_quaterion = action.pick_points[0].translation, action.pick_points[0].quaternion
        pick_right_trans,pick_right_quaterion = action.pick_points[1].translation, action.pick_points[1].quaternion
        place_left_trans,place_left_quaterion = action.place_points[0].translation, action.place_points[0].quaternion 
        place_right_trans,place_right_quaterion = action.place_points[1].translation, action.place_points[1].quaternion
        pick_left = np.concatenate([pick_left_trans,pick_left_quaterion]).tolist()
        pick_right = np.concatenate([pick_right_trans,pick_right_quaterion]).tolist()
        place_left = np.concatenate([place_left_trans,place_left_quaterion]).tolist()
        place_right = np.concatenate([place_right_trans,place_right_quaterion]).tolist()
            
        logger.debug(f"pick_and_place poses:    pick_left:{pick_left}, pick_right:{pick_right}, place_left:{place_left}, place_right:{place_right},")
        exe_result = self.controller.execute_pick_and_place(np.array([pick_left,pick_right,place_left,place_right]))
 
        return exe_result
    
    def _handle_pick_and_place_single(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        if action.pick_points[0] is not None and action.place_points[0] is not None:
            is_left_robot = True
            # use left robot
            pick_trans, pick_quaterion = action.pick_points[0].translation, action.pick_points[0].quaternion
            place_trans, place_quaterion = action.place_points[0].translation, action.place_points[0].quaternion
        elif action.pick_points[1] is not None and action.place_points[1] is not None:
            # use right robot
            is_left_robot = False
            pick_trans,pick_quaterion = action.pick_points[1].translation, action.pick_points[1].quaternion
            place_trans,place_quaterion = action.place_points[1].translation, action.place_points[1].quaternion
        else:
            raise NotImplementedError
        pick = np.concatenate([pick_trans, pick_quaterion]).tolist()
        place = np.concatenate([place_trans, place_quaterion]).tolist()
            
        logger.debug(f"pick_and_place poses:    pick:{pick}, place:{place},")
        exe_result = self.controller.execute_pick_and_place_single(np.array([pick,place]),
                                                                   is_left_robot=is_left_robot)
 
        return exe_result
    
    def _handle_lift(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        joint_values_l, joint_values_r = action.joint_values
        if joint_values_l is not None:
            result = self.controller.execute_lift(joint_values=joint_values_l, robot="l")
        else:
            result = self.controller.execute_lift(joint_values=joint_values_r, robot="r")
        return result
    
    def choose_random_pt(self, virtual_pc_xyz: np.ndarray) -> ActionMessage:
        world_pc_xyz =  transform_point_cloud(virtual_pc_xyz, self.transforms.virtual_to_world_transform)
        xmin = self.option.planning.lift_cloth_action.valid_x_thresh
        valid_index = np.where(world_pc_xyz[:, 0] > xmin)
        world_pc_xyz = world_pc_xyz[valid_index]
        while True:
            random_idx = np.random.choice(world_pc_xyz.shape[0], 1)
            chosen_pt = world_pc_xyz[random_idx].ravel()
            if not self.is_pose_reachable(RigidTransform(translation=chosen_pt.tolist()), is_left_robot=True) and \
                not self.is_pose_reachable(RigidTransform(translation=chosen_pt.tolist()), is_left_robot=False):
                    continue
            chosen_pt[2] += self.option.planning.lift_cloth_action.grasp_waypts_z_offset
            world_pose = pos_rot_to_pose_4x4(tran=chosen_pt, rot=rotation_about_xyz(np.pi, 'y'))
            q_value_l, q_value_r = self.controller.calculate_poses(pose_l=world_pose,pose_r=world_pose, rotation_correction=False)

            if (q_value_l is not None) and (q_value_r is not None):
                if chosen_pt[1] > 0.0:
                    output = ActionMessage(action_type=ActionTypeDef.LIFT, joint_values=[q_value_l, None])
                    return output
                else:
                    output = ActionMessage(action_type=ActionTypeDef.LIFT, joint_values=[None, q_value_r])
                    return output
            
            if q_value_l is not None:
                output = ActionMessage(action_type=ActionTypeDef.LIFT, joint_values=[q_value_l, None])
                return output
            elif q_value_r is not None:
                output = ActionMessage(action_type=ActionTypeDef.LIFT, joint_values=[None, q_value_r])
                return output

    def execute_action(self, action: ActionMessage) -> Optional[ExceptionMessage]:
        if action.action_type == ActionTypeDef.FAIL:
            return ExceptionMessage("Action failed", -1)  
        if action.action_type == ActionTypeDef.FLING:
            return self._handle_fling(action)
        elif action.action_type == ActionTypeDef.DRAG:
            return self._handle_drag_normal(action)
        elif action.action_type == ActionTypeDef.DRAG_HYBRID:
            return self._handle_drag_hybrid(action)
        elif action.action_type == ActionTypeDef.FOLD_1:
            return self._handle_fold_1(action)
        elif action.action_type == ActionTypeDef.FOLD_2:
            return self._handle_fold_2(action)
        elif action.action_type == ActionTypeDef.PICK_AND_PLACE:
            return self._handle_pick_and_place(action)
        elif action.action_type == ActionTypeDef.PICK_AND_PLACE_SINGLE:
            return self._handle_pick_and_place_single(action)
        elif action.action_type == ActionTypeDef.DONE:
            return self._handle_done(action)
        elif action.action_type == ActionTypeDef.HOME:
            return self._handle_home(action)
        elif action.action_type == ActionTypeDef.LIFT:
            return self._handle_lift(action)
        else:
            return NotImplementedError

