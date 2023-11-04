# Error Config for Path Planning.
# Author: Guangfei Zhu

from easydict import EasyDict as edict
from common.datamodels import ExceptionMessage
from enum import Enum


class ErrorCode:
    def __init__(self):
        self.ok = None
        self.robot_lost = ExceptionMessage("robot_lost", 101)
        self.robot_emergency = ExceptionMessage("robot_emergency", 102)
        self.system_fault = ExceptionMessage("system_fault", 103)
        self.plan_failed = ExceptionMessage("plan_failed", 104)
        self.grasp_failed = ExceptionMessage("grasp_failed", 105)
        self.ik_failed = ExceptionMessage("ik_failed", 106)

        self.killed = ExceptionMessage("killed", 201)

        self.empty_results = ExceptionMessage("empty_results", 301)
        self.config_load_failed = ExceptionMessage("config_load_failed", 302)
        self.camera_failed = ExceptionMessage("camera_failed", 401)


error_code = ErrorCode()
