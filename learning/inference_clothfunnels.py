import sys
import os
import os.path as osp

sys.path.append(osp.join('', os.path.dirname(os.path.abspath(__file__))))

from common.space_util import transform_point_cloud
from loguru import logger
from typing import Dict, Tuple, Optional, Union, List, Any
from autolab_core import RigidTransform
from common.datamodels import ActionTypeDef, GarmentTypeDef, ActionMessage, ObservationMessage, ExceptionMessage, \
    PredictionMessage
from manipulation.experiment_real import ExperimentReal
from omegaconf import OmegaConf
from omegaconf import DictConfig
from learning.cloth_funnels.clothfunnels_env import ClothFunnelsEnv
from learning.cloth_funnels.learning_nets import *
from learning.cloth_funnels.visualization_utils import *


class InferenceClothFunnels:
    """
    Inference class for 3D point cloud input with ClothFunnels model
    """
    __VERSION__ = "v1"

    def __init__(
            self,
            model_path: str,
            keypoint_model_path: str,
            experiment: ExperimentReal = None,  # Experiment class
            args: Union[OmegaConf, DictConfig] = None,
            **kwargs):
        self.experiment = experiment

        self.model_path = model_path

        tx_camera_view = np.linalg.inv(self.experiment.transforms.clothfunnels_camera_to_view_transform)
        tx_world_camera = np.linalg.inv(self.experiment.transforms.world_to_clothfunnels_camera_transform)
        cam_intr = self.experiment.transforms.clothfunnels_camera_intrinsics.intrinsic_matrix
        table_low = self.experiment.option.compat.machine.table_limits_low
        table_high = self.experiment.option.compat.machine.table_limits_high

        tx_left_camera = self.experiment.transforms.world_to_left_robot_transform @ tx_world_camera
        tx_right_camera = self.experiment.transforms.world_to_right_robot_transform @ tx_world_camera
        robot_far = self.experiment.option.compat.machine.robot_safe_range[1]
        robot_near = self.experiment.option.compat.machine.robot_safe_range[0]

        img_size = 128
        highres_img_size = 720
        num_rotations = 16
        scales = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0]  # raw
        self.env = ClothFunnelsEnv(tx_camera_view, tx_world_camera, cam_intr,
                                   table_low, table_high, tx_left_camera, tx_right_camera,
                                   robot_far, robot_near,
                                   highres_img_size=highres_img_size,
                                   img_size=img_size,
                                   num_fling_rotations=num_rotations,
                                   num_pick_place_rotations=num_rotations,
                                   scales=scales,
                                   keypoint_model_path=keypoint_model_path)

        ckpt = torch.load(self.model_path, map_location='cuda:0')

        # policy
        deformable_weight = 0.9  # 0.7 -> 0.9 for better performance
        self.policy = MaximumValuePolicy(
            action_primitives=['place', 'fling'],
            num_rotations = num_rotations,
            scale_factors = scales,
            obs_dim = 128,
            pix_grasp_dist = 16,
            pix_drag_dist = 16,
            pix_place_dist = 10,
            deformable_weight = deformable_weight,
            nocs_mode = "collapsed",
            network_gpu = [0, 0],
            action_expl_prob = 0,
            action_expl_decay = 0,
            value_expl_prob = 0,
            value_expl_decay = 0,
            dual_networks = None,
            input_channel_types = 'rgb_pos',
            deformable_pos = True,
            dump_network_inputs=False,
            dump_visualizations=True
        )
        self.policy.load_state_dict(ckpt['net'])

        # folding heuristic
        self.SHIRT_KEYPOINTS_DEF = {
            'left_shoulder': 0,
            'right_shoulder': 1,
            'left_arm_top': 2,
            'right_arm_top': 3,
            'left_bottom_corner': 4,
            'right_bottom_corner': 5,
        }
        self.SHIRT_FOLD1_RULE = [
            ['left_arm_top', 'left_bottom_corner'],
            ['right_arm_top', 'right_bottom_corner'],
        ]
        self.SHIRT_FOLD2_RULE = [
            ['left_shoulder', 'left_bottom_corner'],
            ['right_shoulder', 'right_bottom_corner']
        ]

        # args
        self.args = args
        self.debug = self.args.vis_action

    def predict_from_obs(self, state: dict):
        # resize to 128x128
        transformed_obs = state
        action = self.policy.get_action_single(state, vis=self.debug)
        action_primitives = self.policy.action_primitives

        # Visualizations
        for primitive in action_primitives:
            primitive_max_index = action[primitive]['max_index']
            if primitive_max_index == None:
                print(f"{primitive} not available")
                continue

            primitive_observation_all = transformed_obs[f"transformed_obs_{primitive}"]
            if self.debug:
                save_path = '/home/xuehan/Desktop/CoRL_vis/ClothFunnels'
                from torchvision.utils import make_grid
                rgb_stack = make_grid(primitive_observation_all[:, :3], nrow=len(self.env.scales), padding=2, pad_value=1.)
                rgb_img = (rgb_stack.permute(1, 2, 0).cpu().numpy() * 255.).astype(np.uint8)
                plt.figure(figsize=(10.0, 10.0))
                plt.axis('off')
                plt.imshow(rgb_img)
                plt.title(f'all {primitive} transformed RGB images', fontsize=25)
                plt.savefig(os.path.join(save_path, f'{time.strftime("%Y-%m-%d %H-%M-%S")+" "+str(time.time())}.png'))
                plt.show()

            primitive_observation = primitive_observation_all[primitive_max_index[0]]
            primitive_observation_rgb = primitive_observation[:3, :, :]
            primitive_observation_pos = primitive_observation[(4,5),: :, :]
            rgb_img = np.array(primitive_observation_rgb).transpose(1, 2, 0)
            p = primitive_max_index[1:][::-1]

            rgb_img = (rgb_img * 255).astype(np.uint8)
            bgr_img = rgb_img[:,:,::-1]
            if self.debug:
                save_path = '/home/xuehan/Desktop/CoRL_vis/ClothFunnels'
                plt.figure()
                plt.axis('off')
                plt.imshow(rgb_img)
                plt.title(f'{primitive} transformed RGB image', fontsize=25)
                plt.savefig(os.path.join(save_path, f'{time.strftime("%Y-%m-%d %H-%M-%S")+" "+str(time.time())}.png'))
                plt.show()

            if primitive == "fling":
                p1, p2 = p + np.array([0, 16]),  p + np.array([0, -16])
                p1, p2 = tuple(p1), tuple(p2)
                rgb_img = draw_fling(rgb_img / 255.0, p1, p2)
            elif primitive == "place":
                p1, p2 = p, p  + np.array([0, 10])
                p1, p2 = tuple(p1), tuple(p2)
                rgb_img = draw_place(rgb_img / 255.0, p1, p2)
            if self.debug:
                save_path = '/home/xuehan/Desktop/CoRL_vis/ClothFunnels'
                plt.figure()
                plt.axis('off')
                plt.imshow(rgb_img)
                plt.title(f'{primitive} action prediction', fontsize=25)
                plt.savefig(os.path.join(save_path, f'{time.strftime("%Y-%m-%d %H-%M-%S")+" "+str(time.time())}.png'))
                plt.show()

        primitive = action['best_primitive']
        max_value_index = action['best_index']
        max_value = action['best_value'].item()

        if primitive == "fling":
            left_point, right_point = self.env.pick_and_fling_coord(max_value_index, state[f"{primitive}_info"])
            return dict(left_point=left_point, right_point=right_point, primitive=primitive, max_value=max_value)
        elif primitive == "place":
            start_point, end_point = self.env.pick_and_place_coord(max_value_index, state[f"{primitive}_info"])
            return dict(start_point=start_point, end_point=end_point, primitive=primitive, max_value=max_value)
        else:
            raise NotImplementedError

    def gen_fold_action(self, state: dict, action_type: ActionTypeDef, pc_xyz_world: np.ndarray) \
            -> Tuple[Optional[Dict[str, RigidTransform]], Optional[ExceptionMessage]]:
        """
        generate pick and place points for folding action
        """
        keypoint_world_coords = self.env.get_keypoint_word_coords(state)
        if action_type == ActionTypeDef.FOLD_1:
            fold_rules = self.SHIRT_FOLD1_RULE
        elif action_type == ActionTypeDef.FOLD_2:
            fold_rules = self.SHIRT_FOLD2_RULE
        else:
            raise NotImplementedError

        if self.debug:
            keypoints = state['keypoints']
            img = state['pretransform_observations']
            img = np.ascontiguousarray(img).astype(np.uint8)
            num_keypoints = len(keypoints)

            # Generate 6 random colors
            colors = plt.cm.rainbow(np.linspace(0, 1, num_keypoints))

            # Convert colors to 8-bit RGB for visualization
            colors_8bit = []
            for idx, color in enumerate(colors):
                r, g, b = (np.array(color[:3]) * 255).astype(int)
                color = [r, g, b]
                color = tuple([int(x) for x in color])
                colors_8bit.append(color)
                logger.debug(f'Keypoint {idx} color: {r}, {g}, {b}')

            for idx, coordinate in enumerate(keypoints):
                x, y = coordinate
                cv2.circle(img, (int(x), int(y)), radius=20, color=colors_8bit[idx], thickness=-1)

            plt.title('keypoint')
            plt.imshow(img)
            plt.show()
            save_path = '/home/xuehan/Desktop/CoRL_vis/ClothFunnels'
            plt.savefig(os.path.join(save_path, f'{time.strftime("%Y-%m-%d %H-%M-%S")+" "+str(time.time())}.png'))
            plt.close()

        pick_points = [None, None]
        place_points = [None, None]
        for idx, (begin_pt_name, end_pt_name) in enumerate(fold_rules):
            pick_points[idx] = keypoint_world_coords[self.SHIRT_KEYPOINTS_DEF[begin_pt_name]]
            place_points[idx] = keypoint_world_coords[self.SHIRT_KEYPOINTS_DEF[end_pt_name]]
        # find the closed point to the point cloud
        pick_points[0] = self.find_closest_point(pick_points[0], pc_xyz_world)
        pick_points[1] = self.find_closest_point(pick_points[1], pc_xyz_world)

        poses_world = (RigidTransform(translation=pick_points[0]),
                       RigidTransform(translation=pick_points[1]),
                       RigidTransform(translation=place_points[0]),
                       RigidTransform(translation=place_points[1]))
        # judge whether the predicted action is executable,
        # and transforms it into world-space poses (represented by RigidTransform class)
        transforms, err = self.experiment.is_action_executable(action_type, poses_world, return_detailed_err=False)

        return transforms, err

    def gen_smoothing_action(self,
                             model_output_dict: dict,
                             action_type: ActionTypeDef,
                             pc_xyz_world: np.ndarray) -> \
            Tuple[Optional[Dict[str, RigidTransform]], Optional[ExceptionMessage]]:
        """
        generate pick (place) points for smoothing action (fling or pick_and_place)
        """
        pick_points = [np.zeros(3), np.zeros(3)]
        place_points = [np.zeros(3), np.zeros(3)]
        is_left_robot = None
        if action_type == ActionTypeDef.FLING:
            assert model_output_dict['primitive'] == 'fling'
            pick_points[0] = model_output_dict['left_point']
            pick_points[1] = model_output_dict['right_point']
            # find the closed point to the point cloud
            pick_points[0] = self.find_closest_point(pick_points[0], pc_xyz_world)
            pick_points[1] = self.find_closest_point(pick_points[1], pc_xyz_world)
        elif action_type == ActionTypeDef.PICK_AND_PLACE_SINGLE:
            assert model_output_dict['primitive'] == 'place'
            start_point = model_output_dict['start_point']
            # find the closed point to the point cloud
            dist_all = np.linalg.norm(start_point[np.newaxis, :] - pc_xyz_world, axis=1)
            start_point = pc_xyz_world[np.argmin(dist_all), :]

            end_point = model_output_dict['end_point']
            # assign to robot which is closer to the start point
            left_robot_world_base_pose = self.experiment.transforms.left_robot_base_pos
            right_robot_world_base_pose = self.experiment.transforms.right_robot_base_pos
            left_robot_start_point_dist = np.linalg.norm(start_point - left_robot_world_base_pose)
            right_robot_start_point_dist = np.linalg.norm(start_point - right_robot_world_base_pose)
            # generate fake action for compatability
            pick_point_fake = (self.experiment.transforms.left_robot_base_pos +
                               self.experiment.transforms.right_robot_base_pos) / 2
            pick_point_fake[0] += 0.35  # move fake pick point towards center
            place_point_fake = pick_point_fake.copy()
            if left_robot_start_point_dist < right_robot_start_point_dist:
                pick_points[0], place_points[0] = start_point, end_point
                pick_points[1], place_points[1] = pick_point_fake, place_point_fake
                is_left_robot = True
            else:
                pick_points[0], place_points[0] = pick_point_fake, place_point_fake
                pick_points[1], place_points[1] = start_point, end_point
                is_left_robot = False
        else:
            raise NotImplementedError

        poses_world = (RigidTransform(translation=pick_points[0]),
                       RigidTransform(translation=pick_points[1]),
                       RigidTransform(translation=place_points[0]),
                       RigidTransform(translation=place_points[1]))
        # judge whether the predicted action is executable,
        # and transforms it into world-space poses (represented by RigidTransform class)
        transforms, err = self.experiment.is_action_executable(action_type, poses_world, return_detailed_err=False)

        if action_type == ActionTypeDef.PICK_AND_PLACE_SINGLE:
            # reset invalid action pose to None
            if is_left_robot:
                transforms['pick_right'], transforms['place_right'] = None, None
            else:
                transforms['pick_left'], transforms['place_left'] = None, None
        return transforms, err

    @staticmethod
    def find_closest_point(point: np.ndarray, point_cloud: np.ndarray) -> np.ndarray:
        """
        find the closest point in point cloud to the given point

        Args:
            point: 3D point (3, )
            point_cloud: Nx3 point cloud

        Returns:
            closest point in point cloud
        """
        left_dist_all = np.linalg.norm(point[np.newaxis, :] - point_cloud, axis=1)
        closet_point = point_cloud[np.argmin(left_dist_all), :]
        return closet_point

    def predict_raw_action_type(self, obs_msg: ObservationMessage) -> ActionTypeDef:
        """
        predict raw action type from observation message
        """
        color = obs_msg.projected_rgb_img
        depth = obs_msg.projected_depth_img
        obj_mask = obs_msg.projected_mask_img
        # assert image size is equal to highres_img_size
        highres_img_size = self.env.highres_img_size
        assert color.shape == (highres_img_size, highres_img_size, 3) and \
               depth.shape == (highres_img_size, highres_img_size) and \
               obj_mask.shape == (highres_img_size, highres_img_size)

        # get transformed observation
        state = self.env.get_transformed_obs(color, depth, obj_mask)
        # get action from model prediction
        model_output_dict = self.predict_from_obs(state)
        max_value = model_output_dict['max_value']

        if max_value < 0.15:
            # fully smoothed
            action_type = ActionTypeDef.FOLD_1
        elif model_output_dict['primitive'] == 'fling':
            action_type = ActionTypeDef.FLING
        elif model_output_dict['primitive'] == 'place':
            action_type = ActionTypeDef.PICK_AND_PLACE_SINGLE
        else:
            raise NotImplementedError

        return action_type

    def predict_action(self,
                       obs_msg: ObservationMessage,
                       action_type: ActionTypeDef
                       ) -> Tuple[PredictionMessage, ActionMessage, Optional[ExceptionMessage]]:
        """
        predict action from observation message
        """
        assert action_type is not None, "action_type should not be None"
        color = obs_msg.projected_rgb_img
        depth = obs_msg.projected_depth_img
        obj_mask = obs_msg.projected_mask_img
        # assert image size is equal to highres_img_size
        highres_img_size = self.env.highres_img_size
        assert color.shape == (highres_img_size, highres_img_size, 3) and \
               depth.shape == (highres_img_size, highres_img_size) and \
               obj_mask.shape == (highres_img_size, highres_img_size)

        # get transformed observation
        state = self.env.get_transformed_obs(color, depth, obj_mask)
        # get action from model prediction
        model_output_dict = self.predict_from_obs(state)
        # transform point cloud to world frame
        pc_xyz_world = transform_point_cloud(obs_msg.valid_virtual_pts,
                                             self.experiment.transforms.virtual_to_world_transform)

        if action_type in (ActionTypeDef.FLING, ActionTypeDef.PICK_AND_PLACE_SINGLE):
            # handle smoothing stage
            transforms, err = \
                self.gen_smoothing_action(model_output_dict, action_type, pc_xyz_world)
        elif action_type in (ActionTypeDef.FOLD_1, ActionTypeDef.FOLD_2):
            # handle folding stage
            logger.debug('Folding stage started...')
            transforms, err = self.gen_fold_action(state, action_type, pc_xyz_world)
        elif action_type == ActionTypeDef.DONE:
            # handle done stage
            transforms, err = None, None
        else:
            raise NotImplementedError

        prediction_message = PredictionMessage()
        prediction_message.action_type = action_type
        prediction_message.obs_message = obs_msg

        logger.info(f"action_type={ActionTypeDef.to_string(action_type)}")
        if action_type == ActionTypeDef.DONE:
            return prediction_message, ActionMessage(action_type=action_type,
                                                     garment_type=self.experiment.option.compat.garment_type), None
        elif err is None:
            # normal action
            return (prediction_message, ActionMessage(action_type=action_type,
                                                      garment_type=self.experiment.option.compat.garment_type,
                                                      pick_points=[
                                                          transforms['pick_left'] if 'pick_left' in transforms else None,
                                                          transforms['pick_right'] if 'pick_right' in transforms else None],
                                                      place_points=[
                                                          transforms['place_left'] if 'place_left' in transforms else None,
                                                          transforms[
                                                              'place_right'] if 'place_right' in transforms else None]),
                    None)
        else:
            # abnormal action (can not execute)
            return prediction_message, ActionMessage(action_type=ActionTypeDef.FAIL), \
                ExceptionMessage("Could not find a valid pose.")
