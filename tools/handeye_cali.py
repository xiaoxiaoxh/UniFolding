"""Test FlexivRobot.

Author: Chongzhao Mao
"""

import sys
sys.path.append(".")
import py_cli_interaction
import time
import logging

from planning.lib_wrapper.python.flexiv_robot import FlexivRobot
from planning.controller import DualFliexivController
import numpy as np
import flexivrdk
from planning.configs.config import config 
from utility import parse_pt_states

logger = logging.getLogger("flexiv_cali")
logging.basicConfig(level=logging.INFO)

from typing import List, Tuple, Optional
def read_tcp_position(path: str) -> Tuple[np.ndarray, Optional[Exception]]:
    try:
        with open(path,"r") as f:
            data_str_list = f.readlines()
        data = list(map(lambda data_str: [float(x) for x in data_str.split('\t')], data_str_list))
        # for i in range (len(data)):
        #     data[i] = data[i].strip().split()
        #     for index in range (len(data[i])):
        #         data[i][index] = float(data[i][index])
        data = np.array(data) # [x, y, z, qw, qx, qy, qz]
        data[:, 0:3] /= 1000
        return data, None
    except Exception as e:
        return None, e

def get_gripper_states(gripper: flexivrdk.Gripper) -> dict:
    gripper_states = flexivrdk.GripperStates()
    gripper.getGripperStates(gripper_states)
    return {
        'width': float(gripper_states.width),
        'force': float(gripper_states.force),
        'max_width': float(gripper_states.maxWidth),
        'is_moving': 
            float(gripper_states.isMoving) if hasattr(gripper_states, 'is_moving') else False,
    }
LOCAL_IP = "192.168.2.223"
ROBOT_NAME_CANDIDATES = ["left", "right"]
ROBOT_IP_CANDIDATES = ["192.168.2.100","192.168.2.101"]
POSE_TXT_PATH_CANDIDATES = ["manifests/calibration/poses_left.txt", "manifests/calibration/poses_right.txt"]

def main():
    option = py_cli_interaction.must_parse_cli_sel("select read or write poses.txt", ["read", "write", "gripper", "home"])
    if option == 0:
        robot_sel = py_cli_interaction.must_parse_cli_sel("pose_to_read: ", ROBOT_NAME_CANDIDATES)
        robot_ip = ROBOT_IP_CANDIDATES[robot_sel]
        client=FlexivRobot(robot_ip, LOCAL_IP)
        data, err = read_tcp_position(POSE_TXT_PATH_CANDIDATES[robot_sel])
        print(data)
        while True:
            num = input('Press Enter to change GOAL NUM (Photoneo System):')
            client.move_ptp(data[int(num)-1], 
                    max_jnt_vel=[9, 9, 10, 10, 21, 21, 21],
                    max_jnt_acc=[7.2, 7.2, 8.4, 8.4, 16.8, 16.8, 16.8])

    elif option==1:
        res: List[np.array] = []
        sel = py_cli_interaction.must_parse_cli_sel("pose_to_write: ", ROBOT_NAME_CANDIDATES)
        path_to_write = POSE_TXT_PATH_CANDIDATES[sel]
        robot_ip = ROBOT_IP_CANDIDATES[sel]


        while True:
            _ = input("switch to manual mode, then switch back to auto mode and hit enter to continue")
            client=FlexivRobot(robot_ip, LOCAL_IP)
            array = client.get_tcp_pose()
            print(array)
            array[:3] *= 1000  # convert unit to mm
            res.append(array)

            proceed = py_cli_interaction.must_parse_cli_bool("proceed?", default_value=True)
            if proceed:
                del client
                continue
            else:
                break
        print(res)
        
        with open(path_to_write, "w") as f:
            for entry in res:
                f.write('\t'.join(map(lambda x: str(x), entry.tolist()))+'\n')
        return

    elif option == 2:
        robot_ip = ROBOT_IP_CANDIDATES[py_cli_interaction.must_parse_cli_sel("robot: ", ROBOT_NAME_CANDIDATES)]
        robot = flexivrdk.Robot(robot_ip, LOCAL_IP)
        gripper = flexivrdk.Gripper(robot)
        while True:
            sel = py_cli_interaction.must_parse_cli_sel("select action:", ["close", "open",  "read"])
            if sel ==0:
                gripper.move(0.00, 0.1, 10)
                for _ in range(1000):
                    print(get_gripper_states(gripper))
                    time.sleep(5e-3)
            elif sel == 1:
                gripper.move(0.02, 0.1, 10)
            else:
                print(get_gripper_states(gripper))
    elif option == 3:
        contoller =DualFliexivController(config=config)
        contoller.move_home_with_plan()
        
if __name__ == "__main__":
    main()
