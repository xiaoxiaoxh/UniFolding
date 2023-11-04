import numpy as np
import os.path as osp
import cv2
import open3d as o3d
import tempfile
import random
import os
from scipy.spatial.transform import Rotation as R

def depth_to_point(image_coord, depth, pinhole_camera_intrinsic, extrinsics):
    """Convert a point on a depth map to 3D points in world space"""
    intrinsics = pinhole_camera_intrinsic.intrinsic_matrix
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], \
                     intrinsics[0, 2], intrinsics[1, 2]
    u, v = image_coord
    z = depth
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    pos_camera = np.array([x, y, z, 1.0])
    pos_world = np.linalg.inv(extrinsics) @ pos_camera
    return pos_world[:3]

def point_to_depth_map_location(point, pinhole_camera_intrinsic, extrinsics):
    """Convert a 3D point into u-v coordinate in 2D depth map"""
    intrinsics = pinhole_camera_intrinsic.intrinsic_matrix
    width = pinhole_camera_intrinsic.width
    height = pinhole_camera_intrinsic.height
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], \
        intrinsics[0, 2], intrinsics[1, 2]
    point_in_camera_coord = (extrinsics @ np.concatenate([point, np.array([1.0])])[:, np.newaxis])[:, 0]
    x, y, z = point_in_camera_coord[0], point_in_camera_coord[1], point_in_camera_coord[2]
    u = max(min(round(-(fx * x / z) + cx), width - 1), 0)
    v = max(min(round(fy * y / z + cy), height - 1), 0)
    return u, v

def image_array_to_point_cloud_intrinsic_matrix(image_rgb: np.ndarray, image_depth: np.ndarray, intrinsic: o3d.camera.PinholeCameraIntrinsic, extrinsic_matrix: np.ndarray):
    temp_file_path = osp.join(tempfile.gettempdir(), f'temp_img_{int(random.uniform(10000000,99999999))}.png')

    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_file_path, image_rgb)
    color = o3d.io.read_image(temp_file_path)

    # change .exr format to .png format
    depth_png = (image_depth * 1000).astype(np.uint16)[:, :]
    cv2.imwrite(temp_file_path, depth_png)
    depth = o3d.io.read_image(temp_file_path)

    os.remove(temp_file_path)

    pcd = image_open3d_to_point_cloud_intrinsic_matrix(color, depth, intrinsic, extrinsic_matrix)

    return pcd

def image_open3d_to_point_cloud_intrinsic_matrix(color: o3d.geometry.Image, depth: o3d.geometry.Image, intrinsic: o3d.camera.PinholeCameraIntrinsic, extrinsic_matrix: np.ndarray):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=20, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic, project_valid_depth_only=True)

    # convert to unity space
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # camera to world in unity space
    pcd.transform(np.linalg.inv(extrinsic_matrix))

    return pcd

def transform_matrix_to_position_and_rotation(matrix: np.ndarray):
    position = matrix[:3, 3]
    rotation = R.from_matrix(matrix[:3, :3])
    return position, rotation

def transform_point_cloud(pc: np.ndarray, matrix: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.transform(matrix)
    pc = np.asarray(pcd.points)
    return pc