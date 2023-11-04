import numpy as np
import open3d as o3d
from copy import deepcopy
from common.geometry_util import rigid_transform_3D

def superimpose(current_verts, goal_verts, indices=None):

    current_verts = current_verts.copy()
    goal_verts = goal_verts.copy()

    if indices is not None:
        R, t = rigid_transform_3D(current_verts[indices].T, goal_verts[indices].T)
    else:
        R, t = rigid_transform_3D(current_verts.T, goal_verts.T)

    icp_verts = (R @ current_verts.T + t).T

    return icp_verts
def deformable_distance(goal_verts, current_verts, max_coverage, deformable_weight=0.65, scale=None):
    goal_verts = goal_verts.copy()
    current_verts = current_verts.copy()

    #flatten goals
    goal_verts[:, 2] = 0  # z-axis
    current_verts[:, 2] = 0  # z-axis
    flipped_goal_verts = goal_verts.copy()
    flipped_goal_verts[:, 0] =  -1 * flipped_goal_verts[:, 0]  # x-axis

    real_l2_distance = np.mean(np.linalg.norm(goal_verts - current_verts, axis=1))
    real_l2_distance_flipped = np.mean(np.linalg.norm(flipped_goal_verts - current_verts, axis=1))
    if real_l2_distance_flipped < real_l2_distance:
        real_l2_distance = real_l2_distance_flipped

    #GOAL is RED
    goal_vert_cloud = o3d.geometry.PointCloud()
    goal_vert_cloud.points = o3d.utility.Vector3dVector(goal_verts.copy())
    goal_vert_cloud.paint_uniform_color([1, 0, 0])

    normal_init_vert_cloud = deepcopy(goal_vert_cloud)

    flipped_goal_vert_cloud = o3d.geometry.PointCloud()
    flipped_goal_vert_cloud.points = o3d.utility.Vector3dVector(flipped_goal_verts.copy())
    flipped_goal_vert_cloud.paint_uniform_color([0, 1, 1])

    goal_vert_cloud += flipped_goal_vert_cloud
    #CURRENT is GREEN
    verts_cloud = o3d.geometry.PointCloud()
    verts_cloud.points = o3d.utility.Vector3dVector(current_verts.copy())
    verts_cloud.paint_uniform_color([0, 1, 0])

    THRESHOLD_COEFF = 0.3
    threshold = np.sqrt(max_coverage) * THRESHOLD_COEFF
    #superimpose current to goal
    icp_verts = superimpose(current_verts, goal_verts)
    for i in range(5):
        threshold = THRESHOLD_COEFF * np.sqrt(max_coverage)
        indices = np.linalg.norm(icp_verts - goal_verts, axis=1) < threshold
        icp_verts = superimpose(icp_verts, goal_verts, indices=indices)

    #superimpose reverse goal to current
    reverse_goal_verts = goal_verts.copy()
    R, t = rigid_transform_3D(reverse_goal_verts.T, current_verts.T)
    reverse_goal_verts = (R @ reverse_goal_verts.T + t).T
    indices = np.linalg.norm(reverse_goal_verts - current_verts, axis=1) < threshold
    reverse_goal_verts = superimpose(reverse_goal_verts, current_verts, indices=indices)

    reverse_goal_cloud = o3d.geometry.PointCloud()
    reverse_goal_cloud.points = o3d.utility.Vector3dVector(reverse_goal_verts.copy())
    reverse_goal_cloud.paint_uniform_color([1, 0, 1])

    icp_verts_cloud = o3d.geometry.PointCloud()
    icp_verts_cloud.points = o3d.utility.Vector3dVector(icp_verts.copy())
    icp_verts_cloud.paint_uniform_color([0, 0, 1])

    l2_regular = np.mean(np.linalg.norm(icp_verts - goal_verts, axis=1))
    l2_flipped = np.mean(np.linalg.norm(icp_verts - flipped_goal_verts, axis=1))
    l2_distance = min(l2_regular, l2_flipped)

    icp_distance_regular = np.mean(np.linalg.norm(goal_verts - reverse_goal_verts, axis=1))
    icp_distance_flipped = np.mean(np.linalg.norm(flipped_goal_verts - reverse_goal_verts, axis=1))
    icp_distance = min(icp_distance_regular, icp_distance_flipped)

    #make reward scale invariant
    assert(max_coverage != 0 or scale != 0)
    if scale is None:
        l2_distance /= np.sqrt(max_coverage)
        icp_distance /= np.sqrt(max_coverage)
        real_l2_distance /= np.sqrt(max_coverage)
    else:
        l2_distance /= scale
        icp_distance /= scale
        real_l2_distance /= scale

    weighted_distance = deformable_weight * l2_distance + (1 - deformable_weight) * icp_distance

    weighted_distance = float(weighted_distance)
    l2_distance = float(l2_distance)
    icp_distance = float(icp_distance)
    real_l2_distance = float(real_l2_distance)
    return weighted_distance, l2_distance, icp_distance, real_l2_distance, \
        {"init_vert_cloud": goal_vert_cloud, "normal_init_vert_cloud": normal_init_vert_cloud , "verts_cloud": verts_cloud, 'icp_verts_cloud': icp_verts_cloud, "reverse_init_verts_cloud": reverse_goal_cloud}
