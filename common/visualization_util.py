import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import open3d as o3d
from typing import Optional

def get_vis_idxs(batch_idx, 
        batch_size=None, this_batch_size=None, 
        vis_per_items=1, max_vis_per_epoch=None):
    assert((batch_size is not None) or (this_batch_size is not None))
    if this_batch_size is None:
        this_batch_size = batch_size
    if batch_size is None:
        batch_size = this_batch_size
    
    global_idxs = list()
    selected_idxs = list()
    vis_idxs = list()
    for i in range(this_batch_size):
        global_idx = batch_size * batch_idx + i
        global_idxs.append(global_idx)
        vis_idx = global_idx // vis_per_items
        vis_modulo = global_idx % vis_per_items
        if (vis_modulo == 0) and (vis_idx < max_vis_per_epoch):
            selected_idxs.append(i)
            vis_idxs.append(vis_idx)
    return global_idxs, selected_idxs, vis_idxs

def visualize_pc_and_grasp_points(pc: np.ndarray,
                                  left_pick_point: Optional[np.ndarray] = None,
                                  right_pick_point: Optional[np.ndarray] = None,
                                  left_place_point: Optional[np.ndarray] = None,
                                  right_place_point: Optional[np.ndarray] = None,
                                  grasp_candidates: Optional[np.ndarray] = None,
                                  visualization_pose_dict: Optional[dict] = None,
                                  pc_colors: Optional[np.ndarray] = None,
                                  radius: float = 0.03,
                                  coord_size: float = 0.1):
    """
    Visualize point cloud, grasp points and grasp candidates.

    :param pc: point cloud, (N, 3) np.ndarray
    :param left_pick_point: left pick point, (3,) np.ndarray
    :param right_pick_point: right pick point, (3,) np.ndarray
    :param left_place_point: left place point, (3,) np.ndarray
    :param right_place_point: right place point, (3,) np.ndarray
    :param grasp_candidates: grasp candidates, (N, 3) np.ndarray
    :param visualization_pose_dict: the dict which contains camera pose for Open3D visualization
    :param pc_colors: colors of point cloud, (N, 3) np.ndarray
    :param radius: the radius of grasp points in visualization
    :param coord_size: the size of the coordinate frame
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    if pc_colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(pc_colors)
    else:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_size)
    vis_list = [pcd, coord]
    if left_pick_point is not None and right_pick_point is not None:
        left_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(left_pick_point)
        left_sphere.paint_uniform_color([0.9, 0, 0])  # dark red
        right_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(right_pick_point)
        right_sphere.paint_uniform_color([0, 0, 0.9])  # dark blue
        vis_list.extend([left_sphere, right_sphere])

    if left_place_point is not None and right_place_point is not None:
        left_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(left_place_point)
        left_sphere.paint_uniform_color([0.5, 0.2, 0.2])  # light red
        right_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(right_place_point)
        right_sphere.paint_uniform_color([0.2, 0.2, 0.5])  # light blue
        vis_list.extend([left_sphere, right_sphere])

    # visualize point cloud, pick points and coordinate frame
    if visualization_pose_dict is not None:
        o3d.visualization.draw_geometries(vis_list, **visualization_pose_dict)
    else:
        o3d.visualization.draw_geometries(vis_list)

    if grasp_candidates is not None:
        # visualize all predicted points
        num_candidates = grasp_candidates.shape[0]
        # create a list of colors with num_candidates colors
        grasp_colors = plt.cm.jet(np.linspace(0, 1, num_candidates))[:, :3]
        vis_list = [coord, pcd]
        for idx, candidate in enumerate(grasp_candidates):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(candidate)
            sphere.paint_uniform_color(grasp_colors[idx])
            vis_list.append(sphere)
        # visualize point cloud, coordinate frame and all predicted points
        if visualization_pose_dict is not None:
            o3d.visualization.draw_geometries(vis_list, **visualization_pose_dict)
        else:
            o3d.visualization.draw_geometries(vis_list)