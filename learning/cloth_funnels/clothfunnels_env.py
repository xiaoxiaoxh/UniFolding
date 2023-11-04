from learning.cloth_funnels.transformed_view_env import ImageStackTransformer, get_offset_stack, \
    camera_image_to_view, is_coord_valid_table, is_coord_valid_robot
from learning.cloth_funnels.real_utils import *
from learning.cloth_funnels.learning_utils import generate_coordinate_map
from learning.cloth_funnels.learning_nets import *
from learning.cloth_funnels.geometry import *
from learning.cloth_funnels.visualization_utils import *
from learning.cloth_funnels.keypoint_detector.keypoint_inference import KeypointDetector
import cv2
import torch


class ClothFunnelsEnv:
    def __init__(self, tx_camera_view, tx_world_camera, cam_intr,
                 table_low, table_high, tx_left_camera, tx_right_camera,
                 robot_far, robot_near,
                 highres_img_size=720,
                 img_size=128,
                 num_fling_rotations=16,
                 num_pick_place_rotations=16,
                 scales=(1.25, 1.5, 2.0, 2.25, 2.5, 3.0),
                 keypoint_model_path=None):

        self.tx_camera_view = tx_camera_view
        self.tx_world_camera = tx_world_camera
        self.cam_intr = cam_intr
        self.table_low = table_low
        self.table_high = table_high
        self.tx_left_camera = tx_left_camera
        self.tx_right_camera = tx_right_camera
        self.robot_far = robot_far
        self.robot_near = robot_near
        self.num_fling_rotations = num_fling_rotations
        self.num_pick_place_rotations = num_pick_place_rotations
        self.scales = scales

        # changed code according to simulation codes
        assert self.num_fling_rotations % 4 == 0
        assert self.num_pick_place_rotations % 4 == 0

        all_rotations = np.linspace(-180, 180, max(self.num_pick_place_rotations, self.num_fling_rotations) + 1)

        # Fling rotations range from -90 to 90 degrees
        self.fling_rotations = all_rotations[np.where(np.logical_and(all_rotations >= -90, all_rotations <= 90))]
        self.fling_rotations = self.fling_rotations / 180 * np.pi  # change from deg to rad

        # Pick and place rotations range from -180 to 167.5 degrees
        self.pick_place_rotations = all_rotations[
            np.where(np.logical_and(all_rotations >= -180, all_rotations <= 167.5))]
        self.pick_place_rotations = self.pick_place_rotations / 180 * np.pi  # change from deg to rad

        self.keypoint_detector = KeypointDetector(keypoint_model_path)
        self.img_size = img_size
        assert self.img_size == 128, 'Only support img_size=128 for now'
        self.highres_img_size = highres_img_size

    def transform_points(self, pts, tx_new_old):
        dim = pts.shape[-1]
        assert (dim == 3) or (dim == 2)
        return pts @ tx_new_old[:dim,:dim].T + tx_new_old[:dim,dim]


    def pixel_to_3d(self, depth_im, pix, cam_pose, cam_intr, depth_scale=1):
        pix_y = np.clip(pix[:,0].astype(np.int64), 0, depth_im.shape[0]-1)
        pix_x = np.clip(pix[:,1].astype(np.int64), 0, depth_im.shape[1]-1)
        cam_pts_z = depth_im[pix_y, pix_x]
        cam_pts_z *= depth_scale
        cam_pts_x = (pix[:, 1]-cam_intr[0, 2]) * cam_pts_z/cam_intr[0, 0]
        cam_pts_y = (pix[:, 0]-cam_intr[1, 2]) * cam_pts_z/cam_intr[1, 1]
        cam_pts = np.array([cam_pts_x,cam_pts_y,cam_pts_z]).T
        wrd_pts = transform_points(cam_pts, cam_pose)
        return wrd_pts

    def setup_thread(self, target):
        thread = Thread(target=target)
        thread.daemon = True
        thread.start()
        return thread

    def sharpen_edges(self, rgb, threshold1=0, threshold2=100):
        # kernel = np.ones((2,2), np.uint8)
        edges = cv2.Canny(rgb.transpose(1, 2, 0), threshold1=0, threshold2=100)
        new_image = rgb * np.stack([(255 - edges)/255]*3).astype(np.int32)
        # print("new image shape", new_image.shape)
        return new_image

        # old_nocs_x, old_nocs_y = nocs_from_rgb(pretransform_observation, orn_net)
        # new_nocs_x, new_nocs_y = nocs_from_rgb(new_image.transpose(2, 0, 1), orientation_network=orn_net)

        # fig, axs = plt.subplots(2, 3)

        # axs[0, 0].imshow(pretransform_observation.transpose(1, 2, 0))
        # axs[0, 1].imshow(old_nocs_x)
        # axs[0, 2].imshow(old_nocs_y)

        # axs[1, 0].imshow(new_image)
        # axs[1, 1].imshow(new_nocs_x)
        # axs[1, 2].imshow(new_nocs_y)

    def pltitle(self, arr):
        return f"max:{arr.max():.2f},min:{arr.min():.2f}"

    def visualize_input(self, policy_input):
        random_plots = 10
        random_indices = np.random.choice(np.arange(policy_input.shape[0]), size=random_plots)
        random_indices=np.arange(random_plots)
        fig, axs = plt.subplots(5, random_plots, figsize=(40, 10))
        for ax in axs.flatten():
            ax.set_axis_off()
        fig.tight_layout()
        for i in range(random_plots):
            inp = policy_input[random_indices[i]]
            axs[0, i].set_title(label=pltitle(inp[:3]))
            axs[0, i].imshow(inp[:3].transpose(1, 2, 0))
            axs[1, i].imshow(inp[7])
            axs[1, i].set_title(label=pltitle(inp[7]))
            axs[2, i].imshow(inp[8])
            axs[2, i].set_title(label=pltitle(inp[8]))
            axs[3, i].imshow(inp[-2])
            axs[3, i].set_title(f"min:{inp[-2].min():.2f}, max:{inp[-2].max():.2f}")
            axs[4, i].imshow(inp[-1])
            axs[4, i].set_title(f"min:{inp[-1].min():.2f}, max:{inp[-1].max():.2f}")
        plt.show()

    def get_keypoint_word_coords(self, state: dict):
        keypoints = state['keypoints']

        world_image_stack = ImageStackTransformer(
            img_shape=(self.img_size, self.img_size),
            rotations=[0], scales=[1])
        world_camera = self.tx_world_camera
        camera_view = self.tx_camera_view
        intrinsic = self.cam_intr
        depth_map = state['pretransform_depth']
        world_coords = world_image_stack.get_world_coords_stack(
            depth_map,
            camera_view,
            world_camera,
            intrinsic)

        scale_factor = self.img_size / self.highres_img_size
        keypoint_world_coords = \
            [world_coords[0, int(keypoint[1] * scale_factor), int(keypoint[0] * scale_factor), :] for keypoint in
             keypoints]

        return keypoint_world_coords

    def nocs_from_rgb(self, rgb, orientation_network, network_in_dim=128, n_bins=32):
        """
        Takes in a square rgb image shaped (3, x, x) typed uint8,
        returns tuple of (nocs_x, nocs_y) with shape (128, 128)
        """
        if rgb.shape[:2] != (network_in_dim, network_in_dim):
            rgb = TF.resize(torch.tensor(rgb), network_in_dim)
        network_input_rgb = rgb.unsqueeze(0).float()/255
        out = orientation_network.forward(network_input_rgb).detach()
        nocs_x_bins = out[0, :, 0, :, :]
        nocs_y_bins = out[0, :, 1, :, :]
        mask = (network_input_rgb[0] > 0).sum(dim=0).bool().float()
        nocs_x = torch.argmax(nocs_x_bins, 0) * mask
        nocs_x /= n_bins - 1
        nocs_y = torch.argmax(nocs_y_bins, 0) * mask
        nocs_y /= n_bins - 1

        return nocs_x.cpu().numpy(), nocs_y.cpu().numpy()

    def get_workspace_crop(self, img):
        return img[:, img.shape[1]//2 - img.shape[0]//2: img.shape[1]//2 + img.shape[0]//2, :]

    def get_transformed_obs(self, color, depth, obj_mask):
        COORDINATE_MAP_NORMALIZER=1

        highres_transformer = ImageStackTransformer(
            img_shape=(self.highres_img_size, self.highres_img_size),
            rotations=[0],
            scales=[1]
        )
        tx_camera_view = self.tx_camera_view
        tx = np.eye(3)
        tx[:2, :2] *= (self.highres_img_size / self.img_size)
        highres_tx_camera_view = tx @ tx_camera_view

        highres_observations = highres_transformer.forward_raw(color, highres_tx_camera_view)[0]
        highres_obj_mask = highres_transformer.forward_raw(obj_mask, highres_tx_camera_view)[0]
        highres_obj_mask = np.expand_dims(highres_obj_mask, axis=-1)
        highres_depth = highres_transformer.forward_raw(depth, highres_tx_camera_view)[0]

        env_input = self.get_input(obs_in=(color, depth, obj_mask))

        transformed_obs_dict = {
            'fling_mask':torch.tensor(env_input['pick_and_fling']['is_valid']).bool(),
            'place_mask':torch.tensor(env_input['pick_and_place']['is_valid']).bool(),
            'pretransform_observations':highres_observations,
            'pretransform_mask':highres_obj_mask,
            'pretransform_depth':highres_depth,
            'pretransform_obj_mask':highres_obj_mask,
            'fling_info':env_input['pick_and_fling']['info'],
            'place_info':env_input['pick_and_place']['info'],
            'keypoints':self.keypoint_detector.get_keypoints(highres_observations * highres_obj_mask,
                                                             mask=highres_obj_mask.squeeze(-1)),
        }

        transformed_rgb_place = env_input['pick_and_place']['obs']
        transformed_rgb_fling = env_input['pick_and_fling']['obs']

        transformed_rgb_place = transformed_rgb_place.transpose(0, 3, 1, 2).astype(np.float32)/255
        transformed_rgb_fling = transformed_rgb_fling.transpose(0, 3, 1, 2).astype(np.float32) / 255


        transformer_place = env_input[f'pick_and_place']['info']['transformer']
        transform_tuples_place = transformer_place.transform_tuples

        transformer_fling = env_input[f'pick_and_fling']['info']['transformer']
        transform_tuples_fling = transformer_fling.transform_tuples

        positional_encoding_input_place = np.stack( \
            [generate_coordinate_map(self.img_size, -1*rotation*(360/(2*np.pi)), 1/scale)*COORDINATE_MAP_NORMALIZER for rotation, scale in transform_tuples_place])
        positional_encoding_input_place = positional_encoding_input_place.transpose(0, 3, 1, 2)

        positional_encoding_input_fling = np.stack( \
            [generate_coordinate_map(self.img_size, -1 * rotation * (360 / (2 * np.pi)), 1 / scale) * COORDINATE_MAP_NORMALIZER
             for rotation, scale in transform_tuples_fling])
        positional_encoding_input_fling = positional_encoding_input_fling.transpose(0, 3, 1, 2)

        extra_channels_place = torch.zeros(positional_encoding_input_place.shape[0], 1, positional_encoding_input_place.shape[2], positional_encoding_input_place.shape[3])
        extra_channels_fling = torch.zeros(positional_encoding_input_fling.shape[0], 1,
                                           positional_encoding_input_fling.shape[2],
                                           positional_encoding_input_fling.shape[3])
        transformed_obs_place = torch.tensor(np.concatenate([transformed_rgb_place, extra_channels_place, positional_encoding_input_place], axis=1)).float()
        transformed_obs_fling = torch.tensor(
            np.concatenate([transformed_rgb_fling, extra_channels_fling, positional_encoding_input_fling],
                           axis=1)).float()
        transformed_obs_dict['transformed_obs_place'] = transformed_obs_place
        transformed_obs_dict['transformed_obs_fling'] = transformed_obs_fling

        return transformed_obs_dict


    def transform_obs(self, transformer, depth,
                      obs=None, obj_mask=None):

        world_coords_stack = transformer.get_world_coords_stack(
            depth=depth,
            tx_camera_view=self.tx_camera_view,
            tx_world_camera=self.tx_world_camera,
            cam_intr=self.cam_intr)
        data = {'world_coord': world_coords_stack}
        if obs is not None:
            obs_stack = transformer.forward_raw(obs, self.tx_camera_view)
            data['obs'] = obs_stack
        if obj_mask is not None:
            mask_stack = transformer.forward_raw(
                obj_mask.astype(np.float32), self.tx_camera_view) > 0.5
            data['obj_mask'] = mask_stack
        return data


    def is_coord_valid_robot_mine(self, coords, is_left, robot_far, robot_near):
        tx = self.tx_left_camera if is_left else self.tx_right_camera
        tx_robot_world = tx @ np.linalg.inv(self.tx_world_camera)
        r = is_coord_valid_robot(coords, tx_robot_world,
                                 reach_radius=robot_far, near_radius=robot_near)
        return r

    #
    def get_pick_and_place_input(self, color, depth, obj_mask):
        scales = self.scales
        offset = 10 # move distance in pixels
        # calculating adaptive scaling
        view_mask = camera_image_to_view(obj_mask, self.tx_camera_view,
                                         img_shape=(self.img_size, self.img_size))
        r,c = np.nonzero(view_mask)

        try:
            max_width = max(r.max() - r.min(), c.max() - c.min())
        except:
            print("No max width calculated")
            max_width = 0.5

        adaptive_scale_factor = max_width * 1.5 / self.img_size
        # compute image and coordiante stack
        scales = 1/(np.array(scales) * adaptive_scale_factor)
        transformer = ImageStackTransformer(
            img_shape=(self.img_size, self.img_size),
            rotations=self.pick_place_rotations,
            scales=scales
        )

        data = self.transform_obs(transformer,
                             depth=depth, obs=color, obj_mask=obj_mask)

        obs_stack = data['obs']
        world_coords_stack = data['world_coord']
        obj_mask_stack = data['obj_mask']

        data = self.transform_obs(ImageStackTransformer(
            img_shape=(self.img_size, self.img_size),
            rotations=[0], scales=[1]),
            depth=depth,
            obs=color,
            obj_mask=obj_mask
        )
        center_obs = data['obs'][0]
        center_obj_mask = data['obj_mask'][0]

        # raw validity
        is_table_valid = is_coord_valid_table(world_coords_stack, table_low=self.table_low, table_high=self.table_high)
        # lift point above pick point much be reachable
        coords = world_coords_stack.copy()
        is_left_valid = is_table_valid & self.is_coord_valid_robot_mine(coords, is_left=True, robot_far=self.robot_far,
                                                                        robot_near=self.robot_near)
        is_right_valid = is_table_valid & self.is_coord_valid_robot_mine(coords, is_left=False, robot_far=self.robot_far
                                                                         , robot_near=self.robot_near)
        # since valid set is mostly convex, checking two endpoints are sufficient

        _, is_left_end_valid = get_offset_stack(is_left_valid, offset=offset)
        _, is_right_end_valid = get_offset_stack(is_right_valid, offset=offset)
        is_left_action_valid = is_left_valid & is_left_end_valid
        is_right_action_valid = is_right_valid & is_right_end_valid
        is_start_on_obj = obj_mask_stack

        start_coord = world_coords_stack
        _, end_coord = get_offset_stack(start_coord, offset=offset)

        is_any_valid = is_left_action_valid | is_right_action_valid
        use_left = is_left_action_valid
        obs = obs_stack
        is_valid = is_any_valid

        # set to True
        must_pick_on_obj = True
        if must_pick_on_obj:
            is_valid = is_valid & is_start_on_obj

        info = {
            'pick_on_obj': is_start_on_obj,
            'use_left': use_left,
            'start_coord': start_coord,
            'end_coord': end_coord,
            'center_obs': center_obs,
            'center_obj_mask': center_obj_mask,
            'transformer': transformer,
            'transformed_obj_masks': obj_mask_stack
        }
        return obs, is_valid, info

    def get_pick_and_fling_input(self, color, depth, obj_mask):
        scales = self.scales
        offset = 16 # half width in pixels
        # calculating adaptive scaling
        view_mask = camera_image_to_view(obj_mask, self.tx_camera_view,
                                         img_shape=(self.img_size, self.img_size))
        r,c = np.nonzero(view_mask)

        try:
            max_width = max(r.max() - r.min(), c.max() - c.min())
        except:
            print("No max width calculated")
            max_width = 0.5

        adaptive_scale_factor = max_width * 1.5 / self.img_size
        # compute image and coordiante stack
        scales = 1/(np.array(scales) * adaptive_scale_factor)
        transformer = ImageStackTransformer(
            img_shape=(self.img_size, self.img_size),
            rotations=self.fling_rotations,
            scales=scales
        )
        data = self.transform_obs(transformer,
                             depth=depth, obs=color, obj_mask=obj_mask)
        obs_stack = data['obs']
        world_coords_stack = data['world_coord']
        obj_mask_stack = data['obj_mask']

        data = self.transform_obs(ImageStackTransformer(
            img_shape=(self.img_size, self.img_size),
            rotations=[0], scales=[1]),
            depth=depth,
            obs=color,
            obj_mask=obj_mask
        )
        center_obs = data['obs'][0]
        center_obj_mask = data['obj_mask'][0]

        # raw validty maps
        is_table_valid = is_coord_valid_table(world_coords_stack, table_low=self.table_low, table_high=self.table_high)
        coords = world_coords_stack.copy()
        is_left_valid = is_table_valid & self.is_coord_valid_robot_mine(coords, is_left=True, robot_far=self.robot_far,
                                                                        robot_near=self.robot_near)
        is_right_valid = is_table_valid & self.is_coord_valid_robot_mine(coords, is_left=False, robot_far=self.robot_far
                                                                         , robot_near=self.robot_near)

        is_left_action_valid, _ = get_offset_stack(is_left_valid, offset=offset)
        _, is_right_action_valid = get_offset_stack(is_right_valid, offset=offset)
        is_action_valid = is_left_action_valid & is_right_action_valid
        is_left_on_obj, is_right_on_obj = get_offset_stack(obj_mask_stack, offset=offset)
        is_action_on_obj = is_left_on_obj & is_right_on_obj

        left_coord, right_coord = get_offset_stack(world_coords_stack, offset=offset)
        obs = obs_stack
        is_valid = is_action_valid

        # set to True
        must_pick_on_obj = True
        if must_pick_on_obj:
            is_valid = is_valid & is_action_on_obj

        info = {
            'pick_on_obj': is_action_on_obj,
            'left_coord': left_coord,
            'right_coord': right_coord,
            'center_obs': center_obs,
            'center_obj_mask': center_obj_mask,
            'transformer': transformer,
            'transformed_obj_masks': obj_mask_stack
        }
        return obs_stack, is_valid, info

    def get_input(self, pick_and_place=True, pick_and_fling=True, obs_in=None):

        assert obs_in is not None
        color, depth, obj_mask = obs_in
        obj_mask = obj_mask.astype(bool)
        color[~obj_mask] = 0

        data = dict()
        if pick_and_place:
            obs, is_valid, info = self.get_pick_and_place_input(
                color, depth, obj_mask)
            pp_data = {
                'obs': obs,
                'is_valid': is_valid,
                'info': info
            }
            data['pick_and_place'] = pp_data

        if pick_and_fling:
            obs, is_valid, info = self.get_pick_and_fling_input(
                color, depth, obj_mask)
            pf_data = {
                'obs': obs,
                'is_valid': is_valid,
                'info': info
            }
            data['pick_and_fling'] = pf_data
        return data


    def get_inverse_coord_map(self, shape, transforms):
        identity_map = np.moveaxis(
            np.indices(shape[1:], dtype=np.float32)[::-1],0,-1
        )

        maps = list()
        for tf in transforms:
            tx = np.linalg.inv(tf.params)
            r = transform_points(
                identity_map.reshape(-1,2),
                tx).reshape(identity_map.shape)
            maps.append(r)
        coord_stack = np.stack(maps)
        return coord_stack

    def pick_and_place_coord(self, map_coord, info):
        """
        map_coord: coordinate index into the spatial action map tensor
        """
        start_coord = info['start_coord']
        end_coord = info['end_coord']

        start_point = start_coord[tuple(map_coord)]
        end_point = end_coord[tuple(map_coord)]
        return start_point, end_point

    def pick_and_fling_coord(self, map_coord, info):
        """
        map_coord: coordinate index into the spatial action map tensor
        """
        left_coord = info['left_coord']
        right_coord = info['right_coord']
        left_point = left_coord[tuple(map_coord)]
        right_point = right_coord[tuple(map_coord)]
        return left_point, right_point


