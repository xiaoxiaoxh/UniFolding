from typing import List, Tuple, Union, Optional

import numpy as np
import open3d as o3d

__FAKE_COLORS__ = [
    np.array([0.12156862745098039, 0.4666666666666667, 0.7058823529411765]),
    np.array([1.0, 0.4980392156862745, 0.054901960784313725]),
    np.array([0.17254901960784313, 0.6274509803921569, 0.17254901960784313]),
    np.array([0.8392156862745098, 0.15294117647058825, 0.1568627450980392]),
    np.array([0.5803921568627451, 0.403921568627451, 0.7411764705882353]),
    np.array([0.5490196078431373, 0.33725490196078434, 0.29411764705882354]),
    np.array([0.8901960784313725, 0.4666666666666667, 0.7607843137254902]),
    np.array([0.4980392156862745, 0.4980392156862745, 0.4980392156862745]),
    np.array([0.7372549019607844, 0.7411764705882353, 0.13333333333333333]),
    np.array([0.09019607843137255, 0.7450980392156863, 0.8117647058823529])
]


def visualize_point_cloud_list_with_points(transformed_pcd_list, points: Union[np.ndarray, List[np.ndarray]] = None,
                                           fake_color=False, point_colors=None):
    coordinate = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    transformed_pcd_all = o3d.geometry.PointCloud()
    xyz_list = [np.asarray(pcd.points) for pcd in transformed_pcd_list]
    if fake_color:
        rgb_list = [np.tile(__FAKE_COLORS__[i], (xyz_list[i].shape[0], 1)) for i in range(len(xyz_list))]
    else:
        rgb_list = [np.asarray(pcd.colors) for pcd in transformed_pcd_list]
    transformed_pcd_all.points = o3d.utility.Vector3dVector(np.concatenate(xyz_list, axis=0))
    transformed_pcd_all.colors = o3d.utility.Vector3dVector(np.concatenate(rgb_list, axis=0))
    lookat = np.asarray(transformed_pcd_all.points).mean(axis=0).tolist()

    geometries = [coordinate, transformed_pcd_all]
    if points is not None and len(points) > 0 and all([isinstance(p, np.ndarray) for p in points]):
        for idx, p in enumerate(points):
            sphere = o3d.geometry.TriangleMesh().create_sphere(radius=0.02).translate(p)
            if point_colors is not None and len(point_colors) == len(points):
                sphere.paint_uniform_color(__FAKE_COLORS__[point_colors[idx] % len(__FAKE_COLORS__)])
            else:
                sphere.paint_uniform_color(__FAKE_COLORS__[idx % len(__FAKE_COLORS__)])
            geometries.append(sphere)

    o3d.visualization.draw_geometries(geometries,
                                      front=[0., 0., 1.],
                                      lookat=lookat,
                                      up=[0., 1., 0.],
                                      zoom=0.25)


def pick_n_points_from_pcd(point_cloud: o3d.geometry.PointCloud, n: int) -> Tuple[
    List[np.ndarray], List[np.ndarray], Optional[Exception]]:
    """pick 2 points from point cloud, with interaction

    Args:
        point_cloud (o3d.geometry.PointCloud): point cloud
        n (int): number of points

    Returns:
         Tuple[List[np.ndarray], List[np.ndarray], Optional[Exception]]: points xyz, points index, exception if any
    """
    # coordinate = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # point_cloud += coordinate.pcd
    if n <= 0:
        return [], [], None
    vis1 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window(f'Please press Shift and click {n} Points: Left Right. Press Q to exit.')
    vis1.add_geometry(point_cloud)
    vis1.update_renderer()
    vis1.run()  # user picks points
    vis1.destroy_window()

    pts_sel = vis1.get_picked_points()
    if len(pts_sel) != n:
        return [], [], Exception(f"Please select {n} points")

    pcd_points = np.asarray(point_cloud.points)
    pts = pcd_points[pts_sel]
    return pts, pts_sel, None


def get_live_visualizer():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    return vis
