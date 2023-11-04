import open3d as o3d
import os.path as osp
import numpy as np
import json
from scipy.spatial.transform import Rotation


def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def read_calibration_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_list = []
        for line in lines:
            print(line)
            data = line[:-1].split('\t')
            print(data)
            data = [float(item) for item in data]
            data_list.append(data)
    return np.array(data_list)



if __name__ == '__main__':
    import os
    import py_cli_interaction
    __VERSION_CANDIDATES_DIR_ = 'data/calibration'
    __VERSION_CANDIDATES__ = list(filter(lambda x: osp.isdir(osp.join(__VERSION_CANDIDATES_DIR_, x)), os.listdir(__VERSION_CANDIDATES_DIR_)))
    __VERSION_CANDIDATES__.sort()
    __VERSION__ = __VERSION_CANDIDATES__[py_cli_interaction.must_parse_cli_sel("select calibration version", __VERSION_CANDIDATES__)]

    left_calibration_path = osp.join(__VERSION_CANDIDATES_DIR_, __VERSION__, 'left_robot_camera_cali_matrix.txt')
    right_calibration_path = osp.join(__VERSION_CANDIDATES_DIR_, __VERSION__, 'right_robot_camera_cali_matrix.txt')
    camera_to_left_transform = read_calibration_txt(left_calibration_path)
    camera_to_right_transform = read_calibration_txt(right_calibration_path)
    camera_to_left_transform[:3, -1] = camera_to_left_transform[:3, -1] / 1000
    camera_to_right_transform[:3, -1] = camera_to_right_transform[:3, -1] / 1000
    left_to_camera_transform = np.linalg.inv(camera_to_left_transform)
    right_to_camera_transform = np.linalg.inv(camera_to_right_transform)

    left_z_dir = (left_to_camera_transform[0:3, 0:3] @ np.array([0., 0., 1.])[:, np.newaxis])[:, 0]
    right_z_dir = (right_to_camera_transform[0:3, 0:3] @ np.array([0., 0., 1.])[:, np.newaxis])[:, 0]

    camera_origin = np.array([0., 0., 0.])
    camera_x_dir = np.array([1., 0., 0.])
    camera_y_dir = np.array([0., 1., 0.])
    camera_z_dir = np.array([0., 0., 1.])

    world_z_dir = (left_z_dir + right_z_dir) / 2
    world_z_dir = world_z_dir / np.linalg.norm(world_z_dir)
    left_origin = left_to_camera_transform[:3, -1]
    right_origin = right_to_camera_transform[:3, -1]
    world_origin = (left_origin + right_origin) / 2
    world_y_dir = (left_origin - right_origin) / np.linalg.norm(left_origin - right_origin)
    world_x_dir = np.cross(world_y_dir, world_z_dir)

    world_sample_pts = np.stack([world_origin,
                                 world_origin + world_x_dir,
                                 world_origin + 0.5 * world_x_dir,
                                 world_origin + 10.0 * world_x_dir,
                                 world_origin + world_y_dir,
                                 world_origin + 0.2 * world_y_dir,
                                 world_origin + 7.0 * world_y_dir,
                                 world_origin + world_z_dir,
                                 world_origin + 0.7 * world_z_dir,
                                 world_origin + 3.0 * world_z_dir])
    camera_sample_pts = np.stack([camera_origin,
                                  camera_origin + camera_x_dir,
                                  camera_origin + 0.5 * camera_x_dir,
                                  camera_origin + 10.0 * camera_x_dir,
                                  camera_origin + camera_y_dir,
                                  camera_origin + 0.2 * camera_y_dir,
                                  camera_origin + 7.0 * camera_y_dir,
                                  camera_origin + camera_z_dir,
                                  camera_origin + 0.7 * camera_z_dir,
                                  camera_origin + 3.0 * camera_z_dir])
    world_sample_pts_pcd = o3d.geometry.PointCloud()
    world_sample_pts_pcd.points = o3d.utility.Vector3dVector(world_sample_pts)
    camera_sample_pts_pcd = o3d.geometry.PointCloud()
    camera_sample_pts_pcd.points = o3d.utility.Vector3dVector(camera_sample_pts)

    R, t = rigid_transform_3D(camera_sample_pts.T, world_sample_pts.T)
    print(f"R: {R}, t: {t}")
    world_to_camera_transform = np.concatenate([np.concatenate([R, t], axis=1), np.array([[0., 0., 0., 1.]])], axis=0)
    print(world_to_camera_transform)

    world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5).transform(world_to_camera_transform)
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    left = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(left_to_camera_transform)
    right = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(right_to_camera_transform)
    o3d.visualization.draw_geometries([camera, left, right, world])  # camera coord system

    world_to_left_robot_transform = np.linalg.inv(left_to_camera_transform) @ world_to_camera_transform
    world_to_right_robot_transform = np.linalg.inv(right_to_camera_transform) @ world_to_camera_transform
    world_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    left_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(
        np.linalg.inv(world_to_left_robot_transform))
    right_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(
        np.linalg.inv(world_to_right_robot_transform))
    o3d.visualization.draw_geometries([world_in_world, left_in_world, right_in_world])  # world coord system

    left_rpy_in_world = Rotation.from_matrix(np.linalg.inv(world_to_left_robot_transform)[:3, :3]).as_euler('xyz')
    right_rpy_in_world = Rotation.from_matrix(np.linalg.inv(world_to_right_robot_transform)[:3, :3]).as_euler('xyz')
    print(f'left_rpy_in_world: {left_rpy_in_world}')
    print(f'right_rpy_in_world: {right_rpy_in_world}')

    output_dir = osp.join(__VERSION_CANDIDATES_DIR_, __VERSION__)
    with open(osp.join(output_dir, 'world_to_camera_transform.json'), 'w') as f:
        json.dump(world_to_camera_transform.tolist(), f)
        print('Saving to {}!'.format(osp.join(output_dir, 'world_to_camera_transform.json')))
    with open(osp.join(output_dir, 'world_to_left_robot_transform.json'), 'w') as f:
        json.dump(world_to_left_robot_transform.tolist(), f)
        print('Saving to {}!'.format(osp.join(output_dir, 'world_to_left_robot_transform.json')))
    with open(osp.join(output_dir, 'world_to_right_robot_transform.json'), 'w') as f:
        json.dump(world_to_right_robot_transform.tolist(), f)
        print('Saving to {}!'.format(osp.join(output_dir, 'world_to_right_robot_transform.json')))