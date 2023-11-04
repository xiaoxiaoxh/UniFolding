#!/usr/bin/env python

"""utility.py

Utility methods.
"""

__copyright__ = "Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import math
# pip install scipy
from scipy.spatial.transform import Rotation as R
import numpy as np

def quat2eulerZYX(quat, degree=False):
    """
    Convert quaternion to Euler angles with ZYX axis rotations.

    Parameters
    ----------
    quat : float list
        Quaternion input in [w,x,y,z] order.
    degree : bool
        Return values in degrees, otherwise in radians.

    Returns
    ----------
    float list
        Euler angles in [x,y,z] order, radian by default unless specified otherwise.
    """

    # Convert target quaternion to Euler ZYX using scipy package's 'xyz' extrinsic rotation
    # NOTE: scipy uses [x,y,z,w] order to represent quaternion
    eulerZYX = R.from_quat([quat[1], quat[2],
                            quat[3], quat[0]]).as_euler('xyz', degrees=degree).tolist()

    return eulerZYX



def euler2quat(euler,degrees=False):
    r4 = R.from_euler('zyx', euler, degrees=degrees)
    return r4.as_quat()

def list2str(ls):
    """
    Convert a list to a string.

    Parameters
    ----------
    ls : list
        Source list of any size.

    Returns
    ----------
    str
        A string with format "ls[0] ls[1] ... ls[n] ", i.e. each value 
        followed by a space, including the last one.
    """

    ret_str = ""
    for i in ls:
        ret_str += str(i) + " "
    return ret_str


def parse_pt_states(pt_states, parse_target):
    """
    Parse the value of a specified primitive state from the pt_states string list.

    Parameters
    ----------
    pt_states : str list
        Primitive states string list returned from Robot::getPrimitiveStates().
    parse_target : str
        Name of the primitive state to parse for.

    Returns
    ----------
    str
        Value of the specified primitive state in string format. Empty string is 
        returned if parse_target does not exist.
    """
    for state in pt_states:
        # Split the state sentence into words
        words = state.split()

        if words[0] == parse_target:
            return words[-1]

    return ""


def check_robot_workspace(robot_right_point,robot_left_point):
    """
    check robot_left_point and robot_right_point is in the robot workspace

    Parameters:
    ---------
    robot_left_point: left robot grasp point list
    robot_right_point: right robot grasp poinst list

    Returns
    ---------
    bool 
        result indicate whether the robot point in the robot work space，
        Return True:  free in workspace 
        Return False: no free in workspace
    """
    robot_right_base_point = np.array([0.00115, -0.31, 0.0])
    robot_left_base_point  = np.array([-0.00115, 0.31, 0.0])

    robot_right_distance = math.sqrt((robot_right_point[0:2]-robot_right_base_point[0:2])@
                                     (robot_right_point[0:2]-robot_right_base_point[0:2]).T)
    robot_left_distance  = math.sqrt((robot_left_point[0:2]-robot_left_base_point[0:2])@
                                     (robot_left_point[0:2]-robot_left_base_point[0:2]).T)
    

    if robot_left_distance < 0.35 or robot_left_distance > 0.8  \
       or robot_right_distance < 0.35 or robot_right_distance > 0.8:
         return False
    
    robot_distance = math.sqrt((robot_right_point[0:2]-robot_left_point[0:2])@
                               (robot_right_point[0:2]-robot_left_point[0:2]).T)
    if robot_distance < 0.3:
        return False
    
    robot_x_distance = np.abs(robot_right_point[1]-robot_left_point[1])
    if robot_x_distance < 0.2:
        return False
    
    return True


def make_pseudo_model_action_to_candidates(orient_enable=False,
                                           position_limit=1.0,
                                           high_limit=0.3,
                                           orient_limit = 3.14):
    """
    make the random grasp points
    
    Parameters:
    ---------
    orient_enable: False: make pose without orient rpy
    orient_enable: True:  make pose with orient rpy

    Returns
    ---------
    list:
        right_point,left_point
    """
    candidate_left = np.random.rand(6,)
    candidate_left[0:2] = candidate_left[0:2]*position_limit
    candidate_left[2] = candidate_left[1]*high_limit
    candidate_left[3:6] = candidate_left[3:6]*orient_limit
    
    candidate_right = np.random.rand(6,)
    candidate_right[0:2] = candidate_right[0:2]*position_limit
    candidate_right[1] = -candidate_right[1]
    candidate_right[2] = candidate_right[2]*high_limit
    candidate_right[3:6] = candidate_right[3:6]*orient_limit
    
    if orient_enable == False:
        return np.ndarray([candidate_right[0:3].tolist(),candidate_left[0:3].tolist()])
    else:
        return np.ndarray([candidate_right.tolist(),candidate_left.tolist()])