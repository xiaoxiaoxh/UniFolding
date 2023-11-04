import numpy as np
import cv2
from common.geometry_util import vectorized_meshgrid, vectorized_range

def get_covered_area_from_particles(cloth_particles: np.ndarray, cloth_particle_radius: float) -> float:
    """
    Calculate the covered area by taking max x,y cood and min x,y
    coord, create a discritized grid between the points

    :param cloth_particles: Current positions of the particles in virtual coordinate system
    :param cloth_particle_radius: Radius of the particles
    """
    pos = cloth_particles
    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 1])
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 1])
    init = np.array([min_x, min_y])
    span = np.array([max_x - min_x, max_y - min_y]) / 100.
    pos2d = pos[:, [0, 1]]

    offset = pos2d - init
    slotted_x_low = np.maximum(np.round((offset[:, 0] - cloth_particle_radius) / span[0]).astype(int), 0)
    slotted_x_high = np.minimum(np.round((offset[:, 0] + cloth_particle_radius) / span[0]).astype(int), 100)
    slotted_y_low = np.maximum(np.round((offset[:, 1] - cloth_particle_radius) / span[1]).astype(int), 0)
    slotted_y_high = np.minimum(np.round((offset[:, 1] + cloth_particle_radius) / span[1]).astype(int), 100)
    # Method 1
    grid = np.zeros(10000)  # Discretization
    listx = vectorized_range(slotted_x_low, slotted_x_high)
    listy = vectorized_range(slotted_y_low, slotted_y_high)
    listxx, listyy = vectorized_meshgrid(listx, listy)
    idx = listxx * 100 + listyy
    idx = np.clip(idx.flatten(), 0, 9999)
    grid[idx] = 1

    return np.sum(grid) * span[0] * span[1]

def get_iou_from_2d_points(A: np.ndarray, B: np.ndarray, vis: bool = False):
    min_range = np.min(np.concatenate([np.min(A, axis=0), np.min(B, axis=0)], axis=0), axis=0)
    max_range = np.max(np.concatenate([np.max(A, axis=0), np.max(B, axis=0)], axis=0), axis=0)
    # Normalize the point sets to have x and y coordinates between 0 and 1
    A = (A - min_range) / (max_range - min_range)
    B = (B - min_range) / (max_range - min_range)

    # Scale the normalized point sets to have coordinates between 0 and 99
    A *= 99
    B *= 99

    # Create a blank image with dimensions (100, 100) for each set of points
    mask_A = np.zeros((100, 100), dtype=np.uint8)
    mask_B = np.zeros((100, 100), dtype=np.uint8)

    # Draw the points on the masks as white circles with radius 2
    for point in A:
        cv2.circle(mask_A, tuple(np.int32(point)), 2, 255, -1)
    for point in B:
        cv2.circle(mask_B, tuple(np.int32(point)), 2, 255, -1)

    if vis:
        cv2.imshow('mask A', mask_A)
        cv2.imshow('mask B', mask_B)
        cv2.waitKey()

    # Compute the intersection and union of the masks
    intersection = cv2.bitwise_and(mask_A, mask_B)
    union = cv2.bitwise_or(mask_A, mask_B)

    # Compute the Intersection over Union (IoU) metric
    iou = cv2.countNonZero(intersection) / float(cv2.countNonZero(union))
    return iou