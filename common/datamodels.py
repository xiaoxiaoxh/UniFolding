import dataclasses
from enum import Enum
from typing import Dict, Any, Iterable
from typing import List, Optional

import numpy as np
import omegaconf
import open3d as o3d
from autolab_core import RigidTransform
from rich.console import Console


class ActionTypeDef(Enum):
    FAIL = -1
    FLING = 0
    DRAG = 1
    FOLD_1 = 2
    FOLD_2 = 3
    PICK_AND_PLACE = 4
    DONE = 5
    LIFT = 6
    DRAG_HYBRID = 7  # contact dragging
    PICK_AND_PLACE_SINGLE = 11
    HOME = 100

    @staticmethod
    def from_string(type_str):
        if type_str == 'done':
            return ActionTypeDef.DONE
        elif type_str == 'fail':
            return ActionTypeDef.FAIL
        elif type_str == 'fling':
            return ActionTypeDef.FLING
        elif type_str == 'drag':
            return ActionTypeDef.DRAG
        elif type_str == 'pick_and_place':
            return ActionTypeDef.PICK_AND_PLACE
        elif type_str == 'pick_and_place_single':
            return ActionTypeDef.PICK_AND_PLACE_SINGLE
        elif type_str == 'fold1':
            return ActionTypeDef.FOLD_1
        elif type_str == 'fold2':
            return ActionTypeDef.FOLD_2
        elif type_str == 'drag_hybrid':
            return ActionTypeDef.DRAG_HYBRID
        elif type_str == 'none' or type_str == 'None' or type_str == 'null' or type_str is None:
            return None
        else:
            raise NotImplementedError

    @staticmethod
    def to_string(action_type):
        if action_type == ActionTypeDef.DONE:
            return 'done'
        elif action_type == ActionTypeDef.FAIL:
            return 'fail'
        elif action_type == ActionTypeDef.FLING:
            return 'fling'
        elif action_type == ActionTypeDef.DRAG:
            return 'drag'
        elif action_type == ActionTypeDef.PICK_AND_PLACE:
            return 'pick_and_place'
        elif action_type == ActionTypeDef.PICK_AND_PLACE_SINGLE:
            return 'pick_and_place_single'
        elif action_type == ActionTypeDef.FOLD_1:
            return 'fold1'
        elif action_type == ActionTypeDef.FOLD_2:
            return 'fold2'
        elif action_type == ActionTypeDef.DRAG_HYBRID:
            return 'drag_hybrid'
        elif action_type is None:
            return 'null'
        else:
            raise NotImplementedError(f'unknown action type {action_type}')


class GarmentTypeDef(Enum):
    NONE = 0
    TSHIRT_SHORT = 1
    TSHIRT_LONG = 2

    @staticmethod
    def from_string(type_str):
        if type_str == 'tshirt_short':
            return GarmentTypeDef.TSHIRT_SHORT
        elif type_str == 'tshirt_long':
            return GarmentTypeDef.TSHIRT_LONG
        else:
            raise NotImplementedError

    @staticmethod
    def to_string(garment_type):
        if garment_type == GarmentTypeDef.TSHIRT_SHORT:
            return 'tshirt_short'
        elif garment_type == GarmentTypeDef.TSHIRT_LONG:
            return 'tshirt_long'
        else:
            raise NotImplementedError


class ExecutionErrorTypeDef(Enum):
    NONE = 0
    UNREACHABLE_PICK_LEFT = 1
    UNREACHABLE_PICK_RIGHT = 2
    UNREACHABLE_PICK_DUAL = 3
    UNREACHABLE_PLACE_LEFT = 4
    UNREACHABLE_PLACE_RIGHT = 5
    UNREACHABLE_PLACE_DUAL = 6
    UNSAFE_FOR_DUAL_ARM = 7
    TOO_FAR_FOR_PICK_LEFT = 8
    TOO_FAR_FOR_PICK_RIGHT = 9
    TOO_FAR_FOR_PICK_DUAL = 10
    UNREACHABLE_FOR_DRAG = 11
    TOO_FAR_FOR_PLACE_LEFT = 12
    TOO_FAR_FOR_PLACE_RIGHT = 13
    TOO_FAR_FOR_PLACE_DUAL = 14


@dataclasses.dataclass
class ActionMessage:
    action_type: ActionTypeDef = dataclasses.field(default=ActionTypeDef.DONE)
    garment_type: GarmentTypeDef = dataclasses.field(default=GarmentTypeDef.NONE)
    pick_points: List[RigidTransform] = dataclasses.field(default=(RigidTransform(), RigidTransform()))
    place_points: List[RigidTransform] = dataclasses.field(default=(RigidTransform(), RigidTransform()))
    joint_values: List[np.ndarray] = dataclasses.field(default=(None, None))
    extra_params: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.pick_points is None:
            self.pick_points = [None, None]
        if self.place_points is None:
            self.place_points = [None, None]

        assert len(self.pick_points) == 2
        assert len(self.place_points) == 2
        pass

    @property
    def left_pick_pt(self):
        return self.pick_points[0]

    @left_pick_pt.setter
    def left_pick_pt(self, value: RigidTransform):
        self.pick_points[0] = value

    @property
    def right_pick_pt(self):
        return self.pick_points[1]

    @right_pick_pt.setter
    def right_pick_pt(self, value: RigidTransform):
        self.pick_points[1] = value

    @property
    def left_place_pt(self):
        return self.place_points[0]

    @left_place_pt.setter
    def left_place_pt(self, value: RigidTransform):
        self.place_points[0] = value

    @property
    def right_place_pt(self):
        return self.place_points[1]

    @right_place_pt.setter
    def right_place_pt(self, value: RigidTransform):
        self.place_points[1] = value

    def to_dict(self):
        raise NotImplementedError

    def from_dict(self):
        raise NotImplementedError


def new_action_message() -> ActionMessage:
    return ActionMessage()


@dataclasses.dataclass
class PredictionMessage:
    action_type: ActionTypeDef = dataclasses.field(default=ActionTypeDef.DONE)
    action_iterator: Iterable = dataclasses.field(default_factory=list)
    pc_xyz: np.ndarray = dataclasses.field(default=None)
    attmaps: dict = dataclasses.field(default=None)
    nocs_map: np.ndarray = dataclasses.field(default=None)
    grasp_point_all: np.ndarray = dataclasses.field(default=None)
    grasp_point_nocs_all: np.ndarray = dataclasses.field(default=None)
    virtual_reward_all: np.ndarray = dataclasses.field(default=None)
    real_reward_all: np.ndarray = dataclasses.field(default=None)
    is_safe_to_pick_pair_matrix: np.ndarray = dataclasses.field(default=np.zeros((0, 0), dtype=bool))
    reachable_list: np.ndarray = dataclasses.field(default=np.zeros((0, 2), dtype=bool))
    nn_timing: float = dataclasses.field(default=0.)


@dataclasses.dataclass
class ObservationMessage:
    valid_virtual_pts: np.ndarray = dataclasses.field(default=None)
    valid_virtual_pcd: o3d.geometry.PointCloud = dataclasses.field(default=None)
    raw_virtual_pts: np.ndarray = dataclasses.field(default=None)
    raw_virtual_pcd: o3d.geometry.PointCloud = dataclasses.field(default=None)
    mask_img: np.ndarray = dataclasses.field(default=None)
    rgb_img: np.ndarray = dataclasses.field(default=None)
    projected_rgb_img: np.ndarray = dataclasses.field(default=None)
    projected_depth_img: np.ndarray = dataclasses.field(default=None)
    projected_mask_img: np.ndarray = dataclasses.field(default=None)
    particle_xyz: np.ndarray = dataclasses.field(default=None)
    valid_nocs_pts: np.ndarray = dataclasses.field(default=None)


class ExceptionMessage(Exception):
    def __init__(self, message, code=None):
        super().__init__(self, message)
        self.code = code


if __name__ == '__main__':
    # Built-in examples and tests
    m = new_action_message()
    pass


@dataclasses.dataclass
class AnnotationConfig:
    annotator: str
    root_dir: str
    K: int
    raw_log_namespace: str

    def __post_init__(self) -> None:
        pass


@dataclasses.dataclass
class AnnotationResult:
    annotator: str = dataclasses.field(default='nobody')
    action_type: ActionTypeDef = dataclasses.field(default=ActionTypeDef.FAIL)
    action_poses: List[Optional[np.ndarray]] = dataclasses.field(default_factory=lambda: [None, None, None, None])
    selected_grasp_point_indices: List[List[Optional[int]]] = dataclasses.field(default_factory=list)
    grasp_point_rankings: List[int] = dataclasses.field(default_factory=list)
    fling_gt_is_better_than_rest: Optional[bool] = dataclasses.field(default=None)

    def __post_init__(self) -> None:
        pass

    def to_dict(self) -> dict:
        """to dict for saving

        Returns:
            dict: RESULT

        Desired Format:
        annotation:
            action_type: a # int
            action_poses: [[x, y, z, r, p, y],  # left pick point
                        [x, y, z, r, p, y],  # right pick point
                        [x, y, z, r, p, y],  # left place point (all zeros if non-exists)
                        [x, y, z, r, p, y]]  # right place point (all zeros if non-exists)
            selected_grasp_point_indices:
                - [p1_idx1, p1_idx2, p2_idx1, p2_idx2],  # p1_idx1, p1_idx2 are the indices of the first pair (P1) in the predicted grasp-point list (pose_virtual.prediction.begin). p2_idx1, p2_idx2 are the indices of the second pair (P2) in the predicted grasp-point list (pose_virtual.prediction.begin).
                - ...
            grasp_point_rankings:
                - ranking  # ranking_result [int] is the ranking result (0, 1, 2, 3) -> (>, <, =, ?)
                - ...
        """
        res = {
            'annotator': self.annotator,
            'action_type': self.action_type.value,
            'action_poses': list(map(lambda x: x.tolist() if x is not None else [0., 0., 0., 0., 0., 0.],
                                     self.action_poses)) if self.action_poses is not None else [],
            'selected_grasp_point_indices': self.selected_grasp_point_indices,
            'grasp_point_rankings': self.grasp_point_rankings,
            'fling_gt_is_better_than_rest': self.fling_gt_is_better_than_rest
        }
        return res

    def from_dict(self, d: dict) -> Optional[Exception]:
        try:
            self.annotator = d['annotator'] if 'annotator' in d else 'nobody'
            self.action_type = ActionTypeDef(d['action_type'])
            self.action_poses = [np.array(x) for x in d['action_poses']]
            self.selected_grasp_point_indices = [list(map(lambda x: int(x), entry)) for entry in
                                                 d['selected_grasp_point_indices']]
            self.grasp_point_rankings = d['grasp_point_rankings']
            self.fling_gt_is_better_than_rest = d['fling_gt_is_better_than_rest']
            return None
        except Exception as e:
            return e


class AnnotationContext:
    _curr_pcd: o3d.geometry.PointCloud
    _curr_rgb: np.ndarray
    _entry_name: str
    _raw_log: omegaconf.DictConfig
    vis: o3d.visualization.Visualizer
    annotation_result: AnnotationResult
    console: Console

    def __init__(self, io_module: Optional['AnnotatorIO']) -> None:
        self.annotation_result = AnnotationResult()
        self.io_module = io_module
        pass

    @property
    def entry_name(self) -> str:
        return self._entry_name

    @entry_name.setter
    def entry_name(self, value: str) -> None:
        self._entry_name = value
        self._raw_log, _ = self.io_module.get_raw_log(self._entry_name)
        self._curr_pcd, _ = self.io_module.get_pcd(self._entry_name)
        self._curr_rgb, _ = self.io_module.get_rgb(self._entry_name)

    @property
    def curr_pcd(self) -> o3d.geometry.PointCloud:
        return self._curr_pcd

    @property
    def curr_rgb(self) -> np.ndarray:
        return self._curr_rgb

    @property
    def raw_log(self) -> omegaconf.DictConfig:
        return self._raw_log


class AnnotationFlag(Enum):
    COMPLETED = 1
    UNCOMPLETED = 0
    CORRUPTED = 2
