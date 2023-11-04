import os
import os.path as osp
import sys
from typing import Tuple, List

import cv2
import numpy as np
import open3d as o3d
from loguru import logger

from common.experiment_base import ExperimentRealTransforms

sys.path.append(osp.join(osp.dirname(__file__), "..", '..'))

from harvesters.core import Harvester
from third_party.mvcam.vcamera import vCameraSystem


class PhoXiCamera:
    def __init__(self,
                 dev_id='2020-12-039-LC3',
                 vis=False,
                 external_calibration_path='data/calibration/v4/external_camera_calibration.txt',
                 transforms: ExperimentRealTransforms = None,
                 **kwargs):
        # PhotoneoTL_DEV_<ID>
        self.device_id = "PhotoneoTL_DEV_" + dev_id
        print("--> device_id: ", self.device_id)

        if os.getenv('PHOXI_CONTROL_PATH') is not None:
            self.cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + "/API/lib/photoneo.cti"
        else:
            print('Can not find Phoxi Control PATH!')
            self.cti_file_path = "/API/lib/photoneo.cti"
        print("--> cti_file_path: ", self.cti_file_path)

        self.transforms = transforms
        self.is_trigger_mode_set = False
        self.vis = vis

        external_intrinsics, external_extrinsics, external_distortion_coeff, external_camera_resolution = \
            self.load_calibration_txt(external_calibration_path)
        self.external_intrinsics = external_intrinsics
        self.external_extrinsics = external_extrinsics

        external_cam_sys = vCameraSystem()
        logger.info("--> Load external camera system")
        assert len(external_cam_sys) == 1, 'Can not find any external camera!'
        self.external_camera = external_cam_sys[0]
        self.external_camera.open()


    @staticmethod
    def load_calibration_txt(txt_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load calibration information for external RGB camrea"""
        with open(txt_path, 'r') as f:
            num_list = f.readline().split(' ')[:-1]
            intrinsics = np.array([float(num) for num in num_list], dtype=np.float32).reshape(3, 3)
            num_list = f.readline().split(' ')[:-1]
            distortion_coeff = np.array([float(num) for num in num_list], dtype=np.float32)
            num_list = f.readline().split(' ')[:-1]
            rotation_matrix = np.array([float(num) for num in num_list], dtype=np.float32).reshape(3, 3)
            num_list = f.readline().split(' ')[:-1]
            translation_vector = np.array([float(num) for num in num_list], dtype=np.float32) / 1000.0  # mm -> m
            num_list = f.readline().split(' ')
            camera_resolution = np.array([int(num) for num in num_list], dtype=np.float32)
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotation_matrix
        extrinsics[:3, 3] = translation_vector
        return intrinsics, extrinsics, distortion_coeff, camera_resolution

    @staticmethod
    def colorize_point_cloud(
            pc: np.ndarray,
            rgb_img: np.ndarray,
            external_intrinsics: np.ndarray,
            external_extrinsics: np.ndarray,
    ) -> np.ndarray:
        """
        Colorize point cloud from RGB image of an external camera

        Input:
            pc: point_cloud, (N, 3) float32 ndarray
            rgb_img: 2D RGB image, (H, W, 3) uint8 ndarray
            external_intrinsics: the intrinsic parameters of the external camera
            external_extrinsics: the extrinsic parameters of the external camera
        Returns:
            pc_rgb: colors of point cloud, (N, 3) uint8 ndarray
        """
        width = rgb_img.shape[1]
        height = rgb_img.shape[0]

        # Create transformation matrix from point cloud to external camera
        inverse_trans_mat = np.linalg.inv(external_extrinsics)  # (4, 4)

        # Project point cloud with external camera intrinsic matrix
        pc_hom = np.hstack((pc, np.ones((pc.shape[0], 1), dtype=np.float32)))  # (N, 4)
        pc_camera = (inverse_trans_mat @ pc_hom.T).T  # (N, 4)
        pc_image = (external_intrinsics @ pc_camera[:, :3].T).T  # (N, 3)
        pc_uv = pc_image[:, :2] / pc_camera[:, 2][:, np.newaxis]  # (N, 2)

        # Colorize point cloud
        num_pts = pc.shape[0]
        pc_rgb = np.zeros((num_pts, 3), dtype=np.uint8)  # (N, 3)
        valid_idxs = (pc_uv[:, 0] >= 0) & (pc_uv[:, 0] < width) & (pc_uv[:, 1] >= 0) & (pc_uv[:, 1] < height)  # (N, )
        valid_uv = np.floor(pc_uv[valid_idxs, :]).astype(np.int32)  # (N, 2)
        pc_rgb[valid_idxs, :] = rgb_img[valid_uv[:, 1], valid_uv[:, 0], :]  # (N, 3)
        return pc_rgb

    def start(self):
        self.h = Harvester()
        self.h.add_file(self.cti_file_path, True, True)
        self.h.update()

        # Print out available devices
        print()
        print("Name : ID")
        print("---------")
        for item in self.h.device_info_list:
            print(item.property_dict['serial_number'], ' : ', item.property_dict['id_'])
        print()

        self.ia = self.h.create({'id_': self.device_id})

        self.features = self.ia.remote_device.node_map

        # print(dir(self.features))
        print("TriggerMode BEFORE: ", self.features.PhotoneoTriggerMode.value)
        self.features.PhotoneoTriggerMode.value = "Software"
        print("TriggerMode AFTER: ", self.features.PhotoneoTriggerMode.value)

        # Send every output structure
        self.features.SendTexture.value = True
        self.features.SendPointCloud.value = True
        self.features.SendNormalMap.value = True
        self.features.SendDepthMap.value = True
        self.features.SendConfidenceMap.value = True
        # self.features.SendEventMap.value = True         # MotionCam-3D exclusive
        # self.features.SendColorCameraImage.value = True # MotionCam-3D Color exclusive

    def stop(self):
        logger.info('Closing PhoXi camera!')
        self.ia.stop()
        self.ia.destroy()
        self.h.reset()
        logger.info('Closing External Camera!')
        self.external_camera.close()

    def capture_rgb(self) -> np.ndarray:
        rgb_img = self.external_camera.read()
        if self.vis:
            cv2.imshow('rgb_img', rgb_img)
        return rgb_img

    def capture_pcd(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()

        self.ia.stop()
        self.ia.start()
        # Trigger frame by calling property's setter.
        # Must call TriggerFrame before every fetch.
        self.features.TriggerFrame.execute()  # trigger first frame
        buffer = self.ia.fetch()  # grab first frame
        # print(buffer)

        # self.features.TriggerFrame.execute() # trigger second frame
        # buffer = self.ia.fetch()             # grab second frame
        # print(buffer)

        payload = buffer.payload

        # Order is fixed on the selected output structure. Disabled fields are shown as empty components.
        # Individual structures can enabled/disabled by the following features:
        # SendTexture, SendPointCloud, SendNormalMap, SendDepthMap, SendConfidenceMap, SendEventMap (MotionCam-3D only), SendColorCameraImage (MotionCam-3D Color only)
        # payload.components[#]
        # [0] Texture
        # [1] TextureRGB
        # [2] PointCloud [X,Y,Z,...]
        # [3] NormalMap [X,Y,Z,...]
        # [4] DepthMap
        # [5] ConfidenceMap
        # [6] EventMap
        # [7] ColorCameraImage

        # Texture Image
        texture_component = payload.components[0]
        texture = None
        if texture_component.width > 0 and texture_component.height > 0:
            # Reshape 1D array to 2D array with image size
            texture = texture_component.data.reshape(texture_component.height, texture_component.width,
                                                     1).copy()
            # if self.vis:
            #     texture_screen = cv2.normalize(texture, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
            #     # Show image
            #     cv2.imshow("Texture", texture_screen)

        # Point Cloud
        point_cloud_component = payload.components[2]
        # Normal Map
        norm_component = payload.components[3]
        # Visualize point cloud
        if point_cloud_component.width > 0 and point_cloud_component.height > 0:
            # Reshape for Open3D visualization to N x 3 arrays
            point_cloud = point_cloud_component.data.reshape(
                point_cloud_component.height * point_cloud_component.width, 3).copy()
            norm_map = norm_component.data.reshape(norm_component.height * norm_component.width, 3).copy()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud / 1000.0)
            pcd.normals = o3d.utility.Vector3dVector(norm_map)
            # If texture is present shade points with texture
            if texture is not None:
                color_xyz = np.zeros((point_cloud_component.height * point_cloud_component.width, 3))
                color_xyz[:, 0] = np.reshape(1 / 65536 * texture, -1)
                color_xyz[:, 1] = np.reshape(1 / 65536 * texture, -1)
                color_xyz[:, 2] = np.reshape(1 / 65536 * texture, -1)
                pcd.colors = o3d.utility.Vector3dVector(color_xyz)

            # print(pcd)
            if self.vis:
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                o3d.visualization.draw_geometries([pcd, coord], width=800, height=600)

        return pcd

    @staticmethod
    def fill_holes(depth_img: np.ndarray, plane_depth=1.56):
        new_depth_img = depth_img.copy()
        mask = np.zeros_like(depth_img).astype(np.uint8)
        width, height = depth_img.shape[1], depth_img.shape[0]
        mask[depth_img == 0] = 1
        i_idxs, j_idxs = np.nonzero(mask)[:2]
        i_offsets = np.array([[-1, -1, -1, 0, 1, 1, 1, 0]]).astype(np.int32)
        j_offsets = np.array([[-1, 0, 1, 1, 1, 0, -1, -1]]).astype(np.int32)
        around_pixel_count = i_offsets.shape[1]
        i_around = np.clip(i_idxs[:, np.newaxis] + i_offsets, 0, height - 1)  # (N, 8)
        j_around = np.clip(j_idxs[:, np.newaxis] + j_offsets, 0, width - 1)  # (N, 8)
        depth_around = np.stack([depth_img[i_around[:, idx], j_around[:, idx], 0]
                                 for idx in range(around_pixel_count)], axis=1)  # (N, 8)
        depth_around_mask = (depth_around > 0).astype(np.int32)  # (N, 8)
        valid_pixel_idxs = np.sum(depth_around_mask, axis=1) > 0  # (N, )
        new_depth_img[i_idxs[valid_pixel_idxs], j_idxs[valid_pixel_idxs], 0] = depth_around[valid_pixel_idxs, :].sum(
            axis=1) / \
                                                                               depth_around_mask[valid_pixel_idxs,
                                                                               :].sum(axis=1)
        if plane_depth != 0.:
            new_depth_img[new_depth_img == 0] = plane_depth
        return new_depth_img

    @staticmethod
    def generate_reprojected_rgb_depth_map(pcd: o3d.geometry.PointCloud,
                                           valid_idxs: np.ndarray,
                                           virtual_intrinsics: o3d.camera.PinholeCameraIntrinsic,
                                           virtual_extrinsics: np.ndarray,
                                           plane_point_uvs: np.ndarray,
                                           vis: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Re-project the point cloud into a virtual camera and generate corresponding RGB-D image and mask
        Fill the hols in depth map with plane depth
        Inputs:
            pcd: o3d.geometry.PointCloud, input point cloud
            valid_idxs: np.ndarray (N, ), the array contains foreground indices of the point cloud
            virtual_intrinsics: o3d.camera.PinholeCameraIntrinsic, the intrinsics of the virtual camera
            virtual_extrinsics: np.ndarray (4,4), the extrinsics of the virtual camera
            plane_point_uvs: np.ndarray (M, 2), the uv coordinates of the sample points for calculating table plane depth
        Returns:
            RGB (H, W, 3), depth_map (H, W), mask (H, W)
        """
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        pc_xyz = np.asarray(pcd.points)
        pcd_trans_t = o3d.t.geometry.PointCloud(device)
        pcd_trans_t.point.positions = o3d.core.Tensor(pc_xyz, dtype, device)
        pcd_trans_t.point.colors = o3d.core.Tensor(np.asarray(pcd.colors), dtype, device)
        # generate reprojected RGB-D image
        rgbd_reproj = pcd_trans_t.project_to_rgbd_image(width=virtual_intrinsics.width,
                                                        height=virtual_intrinsics.height,
                                                        intrinsics=virtual_intrinsics.intrinsic_matrix,
                                                        extrinsics=virtual_extrinsics,
                                                        depth_scale=1.0, depth_max=2.0)
        rgb, depth = np.asarray(rgbd_reproj.color.to_legacy()), np.asarray(rgbd_reproj.depth.to_legacy())[..., None]
        # generate mask based on the valid_idxs
        masked_colors = np.zeros((pc_xyz.shape[0], 3))
        masked_colors[valid_idxs, :] = 1.0
        pcd_mask_t = o3d.t.geometry.PointCloud(device)
        pcd_mask_t.point.positions = o3d.core.Tensor(pc_xyz, dtype, device)
        pcd_mask_t.point.colors = o3d.core.Tensor(masked_colors, dtype, device)
        mask_reproj = pcd_mask_t.project_to_rgbd_image(width=virtual_intrinsics.width,
                                                        height=virtual_intrinsics.height,
                                                        intrinsics=virtual_intrinsics.intrinsic_matrix,
                                                        extrinsics=virtual_extrinsics,
                                                        depth_scale=1.0, depth_max=2.0)
        mask = np.asarray(mask_reproj.color.to_legacy())
        # filter RGB image based on mask
        rgb_raw = rgb.copy()
        rgb = rgb * mask
        # fill non-zero depth map with constant
        critical_points_depth = depth[plane_point_uvs[:, 1], plane_point_uvs[:, 0], 0]
        if critical_points_depth.max() - critical_points_depth.min() > 0.5:
            filled_depth_value = critical_points_depth.max()
        else:
            filled_depth_value = critical_points_depth.mean()
        depth[mask[:, :, 0] == 0] = filled_depth_value

        if vis:
            # Apply the colormap to the depth map
            colored_map = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255 / np.max(depth)), cv2.COLORMAP_JET)
            # Display the colored depth map
            cv2.imshow("Depth Map", colored_map)

            # # list to store the coordinates
            # coordinates = []
            # # callback function for mouse, this will be called every time the mouse is left-clicked
            # def select_point(event, x, y, flags, params):
            #     if event == cv2.EVENT_LBUTTONDOWN:
            #         print('Point Selected : ', x, ' ', y)
            #         coordinates.append([x, y])
            #
            # # setup the mouse callback function
            # print('Please select points on the image. Press "c" or "q" to exit.')
            # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
            # cv2.setMouseCallback('image', select_point)
            # # keep looping until the 'q' key is pressed
            # while True:
            #     # display the image and wait for a keypress
            #     cv2.imshow('image', rgb_raw)
            #     key = cv2.waitKey(1) & 0xFF
            #     # if the 'c' key is pressed, break from the loop
            #     if key == ord('c'):
            #         break

            cv2.imshow('RGB', rgb)
            cv2.imshow('mask', mask)
            cv2.waitKey()

        rgb = (rgb * 255).astype(np.uint8)
        mask = mask[:, :, 0]
        depth = depth[:, :, 0]
        return rgb, depth, mask


if __name__ == '__main__':
    camera = PhoXiCamera(vis=True)
    try:
        camera.start()
        for i in range(5):
            print(f'Tring to capture {i}-th point cloud!')
            pcd = camera.capture_pcd()
    finally:
        camera.stop()

