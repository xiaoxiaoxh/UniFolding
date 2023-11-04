import sys
import os
import os.path as osp

sys.path.append(osp.join('', os.path.dirname(os.path.abspath(__file__))))

import time
import open3d as o3d
from loguru import logger
from typing import Dict, Tuple, Optional, Union, List, Any
from autolab_core import RigidTransform
from learning.net.ufo_net import UFONet
from common.datamodels import ActionTypeDef, GarmentTypeDef, ActionMessage, ObservationMessage, ExceptionMessage, PredictionMessage, ExecutionErrorTypeDef
from common.experiment_base import ExperimentBase
# TODO: conditional import ExperimentReal for debugging
# from manipulation.experiment_real import ExperimentReal
from manipulation.experiment_virtual import ExperimentVirtual
from common.visualization_util import visualize_pc_and_grasp_points
from common.pcd_utils import FPS
from common.space_util import transform_point_cloud
import numpy as np
import torch
import MinkowskiEngine as ME

from omegaconf import OmegaConf
from omegaconf import DictConfig


class Inference3D:
    """
    Inference class for 3D point cloud input
    use action iterator for network prediction
    """

    def __init__(
            self,
            model_path: str,
            model_name: str = 'last',
            model_version: str = 'v7',
            experiment: Union[ExperimentBase, ExperimentVirtual] = None,  # Experiment class
            args: Union[OmegaConf, DictConfig] = None,
            **kwargs):
        self.experiment = experiment
        # load model to gpu
        assert model_version in ('v4', 'v5', 'v6', 'v7'), f'model version {model_version} does not exist!'

        checkpoint_dir = osp.join(model_path, 'checkpoints')
        checkpoint_path = osp.join(checkpoint_dir, model_name + '.ckpt')
        model_config = OmegaConf.load(osp.join(model_path, 'config.yaml'))
        # data hyper-params
        self.voxel_size: float = model_config.config.datamodule.voxel_size
        self.num_pc_sample: int = model_config.config.datamodule.num_pc_sample
        self.num_pc_sample_final: int = model_config.config.datamodule.num_pc_sample_final

        logger.info(f'loading model from {checkpoint_path}!')
        if model_version == 'v7':
            model_cpu = UFONet.load_from_checkpoint(checkpoint_path, strict=False)
        else:
            raise NotImplementedError
        self.model_version = model_version

        device = torch.device('cuda:0')
        self.model = model_cpu.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.batch_size = 1

        # args
        self.args = args

    def transform_input(self, pts_xyz: np.ndarray, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # create sparse-tensor batch
        coords, feat = ME.utils.sparse_collate([coords], [feat])

        coords = coords.to(self.model.device)
        feat = feat.to(self.model.device)
        pts_xyz_batch = torch.from_numpy(pc_xyz_slim[sel_pc_idxs, :]).unsqueeze(0).to(self.model.device)
        return pts_xyz_batch, coords, feat

    def transform_output(self, 
                         poses: np.ndarray, 
                         pts_xyz: np.ndarray, 
                         action_type: ActionTypeDef,
                         pts_xyz_raw: np.ndarray = None,
                         fps_pick_override: bool = False) -> \
            Tuple[RigidTransform, RigidTransform, RigidTransform, RigidTransform]:
        """transform and fix output poses"""
        # poses definition:
        # 0: left pick
        # 1: right pick
        # 2: left place
        # 3: right place
        if action_type == ActionTypeDef.FLING:
            if fps_pick_override:
                logger.warning('Use FPS to find grasp points for fling action!')
                # predicted grasp points are too close, use FPS to randomly select grasp points
                fps = FPS(pts_xyz, 2)
                fps.fit()
                selected_pts = fps.get_selected_pts()
                poses[:2, :3] = selected_pts
            poses[:, -1] = -np.pi / 2
        elif action_type == ActionTypeDef.DRAG:
            poses[:, -1] = 0.
        elif action_type == ActionTypeDef.FOLD_1:
            poses[:, -1] = 0.
        elif action_type == ActionTypeDef.FOLD_2:
            poses[:, -1] = -np.pi / 2                        
        elif action_type == ActionTypeDef.PICK_AND_PLACE:
            poses[:, -1] = -np.pi / 2

        pick1 = self.experiment.transforms.virtual_pose_to_world_pose(poses[0])
        pick2 = self.experiment.transforms.virtual_pose_to_world_pose(poses[1])
        place1 = self.experiment.transforms.virtual_pose_to_world_pose(poses[2])
        place2 = self.experiment.transforms.virtual_pose_to_world_pose(poses[3])

        return pick1, pick2, place1, place2

    def predict_raw_action_type(self, obs_msg: ObservationMessage, running_seed: int = None) -> ActionTypeDef:
        "Predict raw action type by the action classifier, this action type could be changed later if required"
        pts_xyz_batch, coords, feat = self.transform_input(obs_msg.valid_virtual_pts, seed=running_seed)
        action_type = self.model.predict_raw_action_type(pts_xyz_batch, coords, feat,
                                                         only_fling_during_smoothing=self.args.only_fling_during_smoothing,
                                                         smoothed_cls_thr=self.args.smoothed_cls_thr)
        return action_type

    # TODO: change pts_xyz_raw to ObservationMessage
    def predict_action(self, obs_message: ObservationMessage, action_type: ActionTypeDef = None,
                       vis: bool = False, running_seed: int = None,
                       ) -> Tuple[PredictionMessage, ActionMessage, Optional[ExceptionMessage]]:
        pts_xyz_raw, pts_xyz_unmasked = obs_message.valid_virtual_pts, obs_message.raw_virtual_pts
        timing = {'start': time.time()}
        pts_xyz_batch, coords, feat = self.transform_input(pts_xyz_raw, seed=running_seed)
        pts_xyz_numpy = pts_xyz_batch[0].cpu().numpy()
        timing['pre_processing'] = time.time() - timing['start']

        assert self.model_version >= 'v6', 'mask is only supported in model version >= v6'
        prediction_message: PredictionMessage = self.model.predict(pts_xyz_batch, coords, feat,
                                                                   action_type=action_type,
                                                                   return_timing=True)
        prediction_message.pc_xyz = pts_xyz_numpy

        timing['nn'] = prediction_message.nn_timing
        timing['nn_prediction_time'] = time.time()

        action_type = prediction_message.action_type if action_type is None else action_type
        logger.info(f"action_type={ActionTypeDef.to_string(action_type)}")
        # enumerate possible actions and check whether the action is executable
        verbose = True
        enable_drag_for_fold2 = False
        if action_type == ActionTypeDef.DONE:
            return prediction_message, ActionMessage(action_type=action_type, 
                                                     extra_params={'score': None, 'timing': timing, 'idxs': [-1, -1]}), None
        elif action_type == ActionTypeDef.DRAG_HYBRID:
            transforms, err = self.experiment.get_drag_hybrid_poses_from_point_cloud(pts_xyz_raw)
            if transforms is not None:
                return (
                    prediction_message,
                    ActionMessage(
                        action_type=action_type,
                        garment_type=self.experiment.option.compat.garment_type,
                        pick_points=[transforms['pick_left'], transforms['pick_right']],
                        place_points=[transforms['place_left'], transforms['place_right']],
                        extra_params={'score': None, 'timing': timing, 'idxs': [-1, -1]}
                    ),
                    None
                )
            else:
                # TODO: handle errors
                raise NotImplementedError
        else:
            enable_drag_for_fold1 = False
            # use model prediction
            pc_xyz_world, poses, transforms, idxs, safe_pair_matrix, reachable_list, action_type = \
                self.filter_model_prediction(prediction_message.action_iterator,
                                             action_type,
                                             pts_xyz_raw,
                                             pts_xyz_numpy,
                                             prediction_message.grasp_point_all)
            prediction_message.is_safe_to_pick_pair_matrix = safe_pair_matrix
            prediction_message.reachable_list = reachable_list

        timing['selection'] = time.time() - timing['nn_prediction_time']

        if transforms is None:
            if verbose:
                logger.warning(f'Could not find a valid pose (end of iterator).')
            if enable_drag_for_fold1:
                return prediction_message, ActionMessage(action_type=ActionTypeDef.FOLD_1,
                                                         extra_params={'score': None, 'timing': timing,
                                                                       'idxs': [-1, -1]}), \
                    ExceptionMessage("The best pose is not valid.")

            elif enable_drag_for_fold2:
                return prediction_message, ActionMessage(action_type=ActionTypeDef.FOLD_2,
                                                         extra_params={'score': None, 'timing': timing,
                                                                       'idxs': [-1, -1]}), \
                    ExceptionMessage("The best pose is not valid.")
            else:
                return prediction_message, ActionMessage(action_type=ActionTypeDef.FAIL,
                                                         extra_params={'score': None, 'timing': timing, 'idxs': [-1, -1]}), \
                    ExceptionMessage("Could not find a valid pose (end of iterator).")
        else:
            logger.info(f'Predict action type : {action_type}!')
            if vis and transforms is not None:
                geometry_list = self.create_vis_geometries(transforms, pc_xyz_world)
                o3d.visualization.draw_geometries(geometry_list,
                                                  lookat=np.array([[0.5, 0., 0.]]).T,
                                                  up=np.array([[1., 0., 0.]]).T,
                                                  front=np.array([[0., 0., 1.]]).T, zoom=1.0)

            timing['total'] = time.time() - timing['start']
            return (
                prediction_message, 
                ActionMessage(
                    action_type=action_type,
                    garment_type=self.experiment.option.compat.garment_type,
                    pick_points=[transforms['pick_left'], transforms['pick_right']],
                    place_points=[transforms['place_left'] if 'place_left' in transforms else None,
                                  transforms['place_right'] if 'place_right' in transforms else None],
                    extra_params={'score': None, 'timing': timing, 'idxs': idxs}
                ),
                None
            )

    def filter_model_prediction(self,
                                action_iterator,
                                action_type: ActionTypeDef,
                                pts_xyz_raw: np.ndarray,
                                pts_xyz_sampled: np.ndarray,
                                grasp_points_all: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray,
            ActionTypeDef]:
        """
        filter all model predictions and find feasible actions
        """
        num_grasp_candidates = grasp_points_all.shape[0]  # K
        assert num_grasp_candidates > 0
        safe_pair_matrix = np.zeros((num_grasp_candidates, num_grasp_candidates)).astype(bool)  # (K, K)
        left_reachable_list = np.zeros((num_grasp_candidates, )).astype(bool)  # (K, )
        right_reachable_list = np.zeros((num_grasp_candidates,)).astype(bool)  # (K, )
        best_poses = None
        best_transforms = None
        best_idxs = None
        has_found_best_pick_pts = False
        if self.args.vis_all_fling_pred and action_type == ActionTypeDef.FLING:
            visualize_pc_and_grasp_points(pts_xyz_sampled, grasp_candidates=grasp_points_all)
        # transform point cloud from virtual space into world space
        pts_xyz_raw_world = transform_point_cloud(pts_xyz_raw, self.experiment.transforms.virtual_to_world_transform)
        ptx_xyz_sampled_world = transform_point_cloud(pts_xyz_sampled, self.experiment.transforms.virtual_to_world_transform)
        # calculate probability for random exploration
        if self.experiment.option.strategy.random_exploration.enable:
            use_random_pred = np.random.random() < \
                                 self.experiment.option.strategy.random_exploration.random_explore_prob
            random_top_ratio = self.experiment.option.strategy.random_exploration.random_explore_top_ratio
            if use_random_pred:
                logger.warning(f'Using random exploration now! Trying to randomly choose from top {random_top_ratio * 100} '
                            f'% action poses...')
        else:
            use_random_pred = False
            random_top_ratio = 0.
        # iterate all possible actions (sorted by scores, large-first)
        for i, (poses, poses_nocs, idxs) in enumerate(action_iterator(use_random_pred, random_top_ratio)):
            idx1, idx2 = idxs  # left pick-point index, right pick-point index            
            poses_world = self.transform_output(poses,
                                                pts_xyz=pts_xyz_sampled,
                                                action_type=action_type,
                                                pts_xyz_raw=pts_xyz_raw)
            if self.args.vis_pred_order and action_type == ActionTypeDef.FLING and i < self.args.vis_pred_order_num:
                open3d_pose_dict = dict(lookat=np.array([[0.5, 0., 0.]]).T, up=np.array([[1., 0., 0.]]).T,
                                        front=np.array([[0., 0., 1.]]).T, zoom=1.0)
                visualize_pc_and_grasp_points(pts_xyz_raw_world,
                                              left_pick_point=poses_world[0].translation,
                                              right_pick_point=poses_world[1].translation,
                                              visualization_pose_dict=open3d_pose_dict)
            # judge whether the predicted action is executable,
            # and transforms it into world-space poses (represented by RigidTransform class)
            transforms, err = self.experiment.is_action_executable(action_type, poses_world, 
                                                                   return_detailed_err=
                                                                   self.args.drag_for_best_fling_pick_pts or self.args.drag_for_fold1 or self.args.drag_for_fold2)

            # judge whether pick poses is reachable
            # TODO: fix bug here, use another for loop to judge fling predictions
            pick1, pick2, place1, place2 = poses_world
            if self.experiment.is_pose_reachable_by_dual_arm(pick1)[0]:
                left_reachable_list[idx1] = True
            if self.experiment.is_pose_reachable_by_dual_arm(pick1)[1]:
                right_reachable_list[idx1] = True
            if self.experiment.is_pose_reachable_by_dual_arm(pick2)[0]:
                left_reachable_list[idx2] = True
            if self.experiment.is_pose_reachable_by_dual_arm(pick2)[1]:
                right_reachable_list[idx2] = True

            # judge whether pose is safe for dual-arm robot
            # TODO: fix bug here, use another for loop to judge fling predictions
            if err is not None and err.code == ExecutionErrorTypeDef.UNSAFE_FOR_DUAL_ARM:
                safe_pair_matrix[idx1, idx2] = False
            else:
                safe_pair_matrix[idx1, idx2] = True

            if err is None and best_transforms is None:
                has_found_best_pick_pts = True
                best_idxs = idxs
                best_poses = poses.copy()
                best_transforms = transforms.copy()
            elif err is not None and not has_found_best_pick_pts:
                if self.experiment.is_pose_safe(pick1, pick2):
                    has_found_best_pick_pts = True
                if has_found_best_pick_pts and self.args.drag_for_best_fling_pick_pts and action_type == ActionTypeDef.FLING \
                    and err.code in (ExecutionErrorTypeDef.TOO_FAR_FOR_PICK_LEFT, ExecutionErrorTypeDef.TOO_FAR_FOR_PICK_RIGHT,
                                ExecutionErrorTypeDef.TOO_FAR_FOR_PICK_DUAL):
                    logger.warning(f'The best pick points for fling action are too far, use drag action instead!')
                    # the pick candidates with best score are too far, enable drag action now
                    best_idxs = [-1, -1]
                    best_poses = None
                    # override action poses
                    best_transforms, err = self.experiment.get_drag_poses_from_target_points(np.stack([
                        transforms['pick_left'].translation, transforms['pick_right'].translation
                    ], axis=0),
                    pc_xyz_world=ptx_xyz_sampled_world)
                    # override action type
                    action_type = ActionTypeDef.DRAG
                elif has_found_best_pick_pts and self.args.drag_for_fold2 and action_type == ActionTypeDef.FOLD_2 \
                        and err.code in (ExecutionErrorTypeDef.TOO_FAR_FOR_PICK_LEFT, 
                                         ExecutionErrorTypeDef.TOO_FAR_FOR_PICK_RIGHT,
                                         ExecutionErrorTypeDef.TOO_FAR_FOR_PICK_DUAL, 
                                         ExecutionErrorTypeDef.TOO_FAR_FOR_PLACE_LEFT, 
                                         ExecutionErrorTypeDef.TOO_FAR_FOR_PLACE_RIGHT,                                         
                                         ExecutionErrorTypeDef.TOO_FAR_FOR_PLACE_DUAL):
                    logger.warning(f'The best pick points for fold2 action are too far, use drag action instead!')
                    # the pick points are too far, enable drag action now
                    # override action poses
                    best_transforms, err = self.experiment.get_drag_hybrid_poses_from_point_cloud(pts_xyz_raw)
                    best_idxs = [-1, -1]
                    best_poses = None
                    # override action type
                    action_type = ActionTypeDef.DRAG_HYBRID
                elif has_found_best_pick_pts and self.args.drag_for_fold1 and action_type == ActionTypeDef.FOLD_1 \
                        and err.code in (ExecutionErrorTypeDef.TOO_FAR_FOR_PLACE_LEFT, ExecutionErrorTypeDef.TOO_FAR_FOR_PLACE_RIGHT,                                         
                                ExecutionErrorTypeDef.TOO_FAR_FOR_PLACE_DUAL):
                    logger.warning(f'The best place points for fold1 action are too far, use drag action instead!')
                    # the pick points are too far, enable drag action now
                    # override action poses
                    best_transforms, err = self.experiment.get_drag_hybrid_poses_from_point_cloud(pts_xyz_raw)
                    best_idxs = [-1, -1]
                    best_poses = None
                    # override action type
                    action_type = ActionTypeDef.DRAG_HYBRID

        reachable_matrix = np.stack([left_reachable_list, right_reachable_list], axis=1)  # (K, 2)
        if self.args.debug and action_type == ActionTypeDef.FLING:
            logger.debug(f'fling reachable matrix: {reachable_matrix}')

        if best_transforms is None:
            if action_type == ActionTypeDef.FLING:
                # can't find any valid pick poses for fling action, use FPS to find pick-points instead
                for i, (poses, poses_nocs, idxs) in enumerate(action_iterator()):
                    poses_world = self.transform_output(poses,
                                                        pts_xyz=pts_xyz_sampled,
                                                        action_type=action_type,
                                                        fps_pick_override=True)
                    # judge whether the predicted action is executable,
                    # and transforms it into world-space poses (represented by RigidTransform class)
                    transforms, err = self.experiment.is_action_executable(action_type, poses_world)

                    # record the best transforms
                    if err is None and best_transforms is None:
                        best_idxs = np.ones_like(idxs) * -1  # no valid idxs
                        best_poses = poses.copy()
                        best_transforms = transforms.copy()
                        break
            elif action_type != ActionTypeDef.DONE and self.args.vis_err_actin:
                logger.debug(f'predicted poses: {transforms}')
                logger.error('Failed to find a valid action! Show visualization for debugging...')
                # visualize action poses even if they are un-executable
                geometry_list = self.create_vis_geometries(transforms, pts_xyz_raw_world)
                o3d.visualization.draw_geometries(geometry_list,
                                                lookat=np.array([[0.5, 0., 0.]]).T,
                                                up=np.array([[1., 0., 0.]]).T,
                                                front=np.array([[0., 0., 1.]]).T, zoom=1.0)
            else:
                if self.args.debug:
                    logger.debug(f'predicted poses: {best_transforms}')
        return pts_xyz_raw_world, best_poses, best_transforms, best_idxs, safe_pair_matrix, reachable_matrix, action_type

    def create_vis_geometries(self, transforms: dict, pc_xyz_world: np.ndarray, pc_offset: tuple = (0., 0., 0.)):
        input_pcd = o3d.geometry.PointCloud()
        input_pcd.points = o3d.utility.Vector3dVector(pc_xyz_world)
        input_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pc_xyz_world) * 0.5)

        world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        left_robot = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        left_robot.transform(self.experiment.transforms.left_robot_to_world_transform)
        right_robot = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        right_robot.transform(self.experiment.transforms.right_robot_to_world_transform)

        grasp_point_list = []
        dir_point_list = []
        for key, transform in transforms.items():
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025).translate(transform.translation)
            if key == 'pick_left':
                sphere.paint_uniform_color([0.9, 0.0, 0.0])  # dark red
            elif key == 'pick_right':
                sphere.paint_uniform_color([0., 0.0, 0.9])  # dark blue
            elif key == 'place_left':
                sphere.paint_uniform_color([0.5, 0.2, 0.2])  # light red
            elif key == 'place_right':
                sphere.paint_uniform_color([0.2, 0.2, 0.5])  # light blue
            grasp_point_list.append(sphere)

            # theta = transform.euler_angles[-1]
            # grasp_point_dir = np.array([math.cos(-theta), math.sin(-theta), 0])
            # start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015). \
            #     translate(transform.translation - grasp_point_dir * 0.03)
            # start_sphere.paint_uniform_color([0., 1., 0.])  # green
            # end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015). \
            #     translate(transform.translation + grasp_point_dir * 0.03)
            # end_sphere.paint_uniform_color([0., 0., 0.])  # black
            # dir_point_list.extend([start_sphere, end_sphere])

        geometry_list = [world, left_robot, right_robot, input_pcd] + grasp_point_list + dir_point_list
        # add offset for all geometries (only for visualization)
        geometry_list = [geometry.translate(pc_offset) for geometry in geometry_list]
        return geometry_list
