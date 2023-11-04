import time
from typing import Optional, Tuple

import yaml
from pydantic import BaseModel
from statemachine import State, StateMachine
import random
from loguru import logger

from common.datamodels import ActionMessage, ActionTypeDef, ExceptionMessage, ObservationMessage, PredictionMessage
from common.registry import ExperimentRegistry
from common.statemachine import GarmentStateDef
from common.logging_utils import Logger as ExpLogger
from common.visualization_util import visualize_pc_and_grasp_points
from planning.error_config import error_code
from tools.debug_controller import get_remote_action_type_str


class GarmentMachineConditions(BaseModel):
    garment_operable: Optional[bool] = None
    garment_reachable: Optional[bool] = None
    garment_smooth_enough: Optional[bool] = None
    garment_need_drag: Optional[bool] = None
    garment_fling_drag_success: Optional[bool] = None
    garment_folded_once_success: Optional[bool] = None
    garment_folded_twice_success: Optional[bool] = None


class GarmentStateMachine(StateMachine):
    """
    Garment StateMachine
    """
    unknown = State(name=GarmentStateDef.UNKNOWN, initial=True)
    crumpled = State(name=GarmentStateDef.CRUMPLED)
    unreachable = State(name=GarmentStateDef.UNREACHABLE)
    smoothed = State(name=GarmentStateDef.SMOOTHED)
    folded_once = State(name=GarmentStateDef.FOLDED_ONCE)
    folded_twice = State(name=GarmentStateDef.FOLDED_TWICE)
    success = State(name=GarmentStateDef.SUCCESS, final=True)
    failed = State(name=GarmentStateDef.FAILED, final=True)

    # transitions
    begin = (
            unknown.to(crumpled, cond=["observation_is_valid", "garment_operable", "garment_reachable"])
            | unknown.to(unreachable, unless="garment_reachable")
            | unknown.to(failed, cond=["observation_is_valid"], unless="garment_operable")
            | unknown.to(unknown)
    )

    fling_drag = (
            crumpled.to(smoothed, cond=["garment_smooth_enough", "garment_operable"])
            | crumpled.to(unreachable, unless="garment_reachable")
            | crumpled.to(failed, cond="garment_step_threshold_exceeded", unless="garment_operable")
            | crumpled.to(crumpled)
    )

    lift = (
        unreachable.to(crumpled, cond="garment_reachable")
        | unreachable.to(unreachable, unless="garment_reachable")
    )

    fold_once = (
            smoothed.to(folded_once, cond=["garment_folded_once_success", "garment_operable"])
            | smoothed.to(unreachable, unless="garment_reachable")
            | smoothed.to(failed, cond="garment_step_threshold_exceeded", unless="garment_operable")
            | smoothed.to(crumpled, unless="garment_smooth_enough")
            | smoothed.to(smoothed)
    )

    fold_twice = (
            folded_once.to(folded_twice, cond=["garment_folded_twice_success", "garment_operable"])
            | folded_once.to(unreachable, unless="garment_reachable")
            | folded_once.to(failed, cond="garment_step_threshold_exceeded", unless="garment_operable")
            | folded_once.to(crumpled, unless="garment_smooth_enough")
            | folded_once.to(folded_once)
    )

    end = (
            folded_twice.to(unreachable, unless="garment_reachable")
            | folded_twice.to(failed, cond="garment_step_threshold_exceeded")
            | folded_twice.to(success)
    )

    loop = (
            begin
            | lift
            | fling_drag
            | fold_once
            | fold_twice
            | end
    )

    def __init__(
            self,
            disp: bool = False,
    ):
        self.step_idx = 0
        self._disp: bool = disp
        _r = ExperimentRegistry()
        self.max_num_steps = _r.cfg.experiment.strategy.step_num_per_trial

        self.condition: GarmentMachineConditions = GarmentMachineConditions()
        self._latest_err: Optional[ExceptionMessage] = None
        self._latest_logger: Optional[ExpLogger] = None
        self._latest_observation: Optional[ObservationMessage] = None
        self._latest_inference: Optional[PredictionMessage] = None
        self._latest_action: Optional[ActionMessage] = None

        self._initialized = False
        super().__init__(allow_event_without_transition=True)
        self._initialized = True

    def observation_is_valid(self) -> bool:
        return self._latest_observation is not None

    def garment_operable(self) -> bool:
        return self.condition.garment_operable

    def garment_reachable(self) -> bool:
        return self.condition.garment_reachable

    def garment_smooth_enough(self) -> bool:
        return self.condition.garment_smooth_enough

    def garment_need_drag(self) -> bool:
        return self.condition.garment_need_drag

    def garment_fling_drag_success(self) -> bool:
        return self.condition.garment_fling_drag_success

    def garment_folded_once_success(self) -> bool:
        return self.condition.garment_folded_once_success

    def garment_folded_twice_success(self) -> bool:
        return self.condition.garment_folded_twice_success

    def garment_step_threshold_exceeded(self) -> bool:
        return self.step_idx > self.max_num_steps

    def on_exit_unknown(self):
        _r = ExperimentRegistry()
        logger.info("stage 2: randomize the garment")
        if _r.cfg.experiment.strategy.random_lift_in_each_trial:
            self._try_lift_to_center()

    def on_enter_unknown(self):
        _r = ExperimentRegistry()
        logger.info("Starting Episode {}, Object {}, Trial {}".format(_r.episode_idx, _r.garment_id, _r.trial_idx))
        # reset the robot to the home position
        _r.exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))

        # protected section
        self._latest_observation, err = _r.exp.capture_pcd()
        self.update_condition()

    def _capture(self):
        self._latest_observation, err = ExperimentRegistry().exp.capture_pcd()

    def _init_logger(self):
        _r = ExperimentRegistry()
        cfg = _r.cfg
        exp = _r.exp
        self._latest_logger = ExpLogger(
            namespace=cfg.logging.namespace, config=cfg.logging, tag=cfg.logging.tag
        )
        self._latest_logger.init()
        self._latest_logger.log_running_config(cfg)
        self._latest_logger.log_commit(cfg.experiment.environment.project_root)
        self._latest_logger.log_model(
            cfg.inference.model_path, cfg.inference.model_name
        )
        self._latest_logger.log_calibration(
            exp.transforms.camera_to_world_transform,
            exp.transforms.left_robot_to_world_transform,
            exp.transforms.right_robot_to_world_transform,
        )
        self._latest_logger.log_garment_id(_r.garment_id)
        self._latest_logger.log_episode_idx(_r.episode_idx)
        self._latest_logger.log_trial_idx(_r.trial_idx)
        self._latest_logger.log_action_step(self.step_idx)

    def _try_lift_to_center(self, n_try: int = 3):
        _r = ExperimentRegistry()
        _failed = True
        while _failed:
            if _r.cfg.experiment.compat.use_real_robots:
                joint_values_message = _r.exp.choose_random_pt(self._latest_observation.valid_virtual_pts)
                logger.info("Garment UNREACHABLE, Implement LIFT action!")
                _r.exp.execute_action(action=joint_values_message)
            else:
                raise NotImplementedError
            self._latest_observation, self._latest_err = _r.exp.capture_pcd()

            if self._latest_err is not None:
                self.current_state = GarmentStateDef.FAILED
                return ExceptionMessage("Capture failed")
            elif n_try == 0:
                return ExceptionMessage("Lift retry run out")
            else:
                _failed = not _r.exp.is_garment_reachable(self._latest_observation.mask_img)

            n_try -= 1

        return None

    def _get_observation(self) -> Tuple[ObservationMessage, Optional[Exception]]:
        _r = ExperimentRegistry()
        if _r.cfg.experiment.strategy.check_grasp_failure_before_action and not _r.exp.is_garment_reachable(self._latest_observation.mask_img):
            self._try_lift_to_center()
        else:
            self._latest_observation, self._latest_err = _r.exp.capture_pcd()
            return self._latest_observation, self._latest_err

    def _log_before_action(self, obs: ObservationMessage):
        self._latest_logger.log_pcd_raw("begin", obs.raw_virtual_pcd, only_npz=True)
        self._latest_logger.log_rgb("begin", obs.rgb_img)
        self._latest_logger.log_mask("begin", obs.mask_img)
        self._latest_logger.log_pcd_processed("begin", obs.valid_virtual_pcd, only_npz=True)

    def _log_after_prediction(self, p: PredictionMessage, a: ActionMessage):
        _r = ExperimentRegistry()
        self._latest_logger.log_pose_prediction_virtual(
            "begin", p.grasp_point_all
        )
        self._latest_logger.log_decision("begin", a.extra_params["idxs"])
        self._latest_logger.log_action_type(
            ActionTypeDef.to_string(a.action_type)
        )
        left_pick_point_in_virtual, right_pick_point_in_virtual = _r.exp.get_pick_points_in_virtual(a)
        left_place_point_in_virtual, right_place_point_in_virtual = _r.exp.get_place_points_in_virtual(a)

        self._latest_logger.log_pose_gripper_virtual(
            "begin", left_pick_point_in_virtual, right_pick_point_in_virtual
        )
        self._latest_logger.log_pose_gripper_virtual(
            "begin", left_place_point_in_virtual, right_place_point_in_virtual
        )
        self._latest_logger.log_predicted_reward(
            "virtual", p.virtual_reward_all
        )
        self._latest_logger.log_predicted_reward(
            "real", p.real_reward_all
        )
        self._latest_logger.log_reachable_matrix(p.reachable_list)
        self._latest_logger.log_safe_to_pick_matrix(
            p.is_safe_to_pick_pair_matrix
        )

    def _get_action_type(self) -> ActionTypeDef:
        _r = ExperimentRegistry()
        if _r.cfg.inference.args.action_type_override.type is not None:
            _action_type = _r.cfg.inference.args.action_type_override.type
        else:
            _action_type = "null"
            if _r.debug_client is not None:
                _action_type = get_remote_action_type_str(_r.debug_client)
            if _action_type == 'null':
                if self.current_state.value == GarmentStateDef.CRUMPLED:
                    _action_type = ActionTypeDef.to_string(ActionTypeDef.FLING)
                elif self.current_state.value == GarmentStateDef.SMOOTHED:
                    _action_type = ActionTypeDef.to_string(ActionTypeDef.FOLD_1)
                elif self.current_state.value == GarmentStateDef.FOLDED_ONCE:
                    _action_type = ActionTypeDef.to_string(ActionTypeDef.FOLD_2)
                elif self.current_state.value == GarmentStateDef.FOLDED_TWICE:
                    _action_type = ActionTypeDef.to_string(ActionTypeDef.DONE)
                else:
                    _action_type = ActionTypeDef.to_string(ActionTypeDef.FAIL)
                    raise Exception("TODO")
        return ActionTypeDef.from_string(_action_type)

    def _abort_with_error(self, err):
        _r = ExperimentRegistry()
        logger.warning(f'{err}')
        self._latest_logger.finalize()
        self._latest_logger.close()
        self._latest_logger = None
        self._latest_err = err
        self.current_state = self.failed
        _r.exp.controller.actuator.open_gripper()
        if not _r.cfg.experiment.strategy.skip_all_errors:
            raise err
        else:
            return None

    def _execute_action_failsafe(self, a: ActionMessage) -> Optional[ExceptionMessage]:
        _r = ExperimentRegistry()
        err = None
        if a.action_type not in (ActionTypeDef.DONE, ActionTypeDef.FAIL):
            err = _r.exp.execute_action(a)
        elif a.action_type == ActionTypeDef.DONE:
            logger.warning(f"Task done! Skipping action now...")
        elif a.action_type == ActionTypeDef.FAIL:
            if _r.cfg.experiment.strategy.skip_all_errors:
                logger.warning('Skipping ActionTypeDef.FAIL...')
                err = ExceptionMessage(
                    code=error_code.plan_failed,
                    message='ActionTypeDef.FAIL...',
                )

        if err is not None:
            # execution failed
            _r.exp.controller.actuator.open_gripper()
            _r.exp.controller.move_home_with_plan()
            if not _r.cfg.experiment.strategy.skip_all_errors:
                raise err
        return err

    def _debug_visualize_points(self, a: ActionMessage, o: ObservationMessage):
        _r = ExperimentRegistry()
        left_pick_point_in_virtual, right_pick_point_in_virtual = _r.exp.get_pick_points_in_virtual(a)
        left_place_point_in_virtual, right_place_point_in_virtual = _r.exp.get_place_points_in_virtual(a)

        visualize_pc_and_grasp_points(
            o.raw_virtual_pts,
            left_pick_point=left_pick_point_in_virtual[:3],
            right_pick_point=right_pick_point_in_virtual[:3],
        )

        visualize_pc_and_grasp_points(
            o.raw_virtual_pts,
            left_pick_point=left_place_point_in_virtual[:3],
            right_pick_point=right_place_point_in_virtual[:3],
        )

    def _finalize_after_action(self, a: ActionMessage, err: ExceptionMessage):
        _r = ExperimentRegistry()
        if not _r.cfg.experiment.compat.only_capture_pcd_before_action:
            logger.info("stage 3.4: capture pcd after action")
            _r.exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))
            obs, _ = _r.exp.capture_pcd()
            self._latest_logger.log_pcd_raw("end", obs.raw_virtual_pcd, only_npz=True)
            self._latest_logger.log_rgb("end", obs.rgb_img)
            self._latest_logger.log_mask("end", obs.mask_img)
            self._latest_logger.log_pcd_processed("end", obs.valid_virtual_pcd, only_npz=True)

        try:
            self._latest_logger.finalize()
            self._latest_logger.close()
        except yaml.representer.RepresenterError as e:
            logger.error(e)
            logger.debug(self._latest_logger._metadata)

        if (a.action_type in (ActionTypeDef.DONE, ActionTypeDef.FAIL)
                or (err is not None and err.code != error_code.grasp_failed)):
            # early stopping for this trial
            self.current_state = self.failed
            return

    def _action_loop(
            self
    ) -> Tuple[
        Optional[ObservationMessage],
        Optional[ActionMessage],
        Optional[PredictionMessage],
        Optional[ExceptionMessage]
    ]:
        _r = ExperimentRegistry()
        cfg, exp = _r.cfg, _r.exp
        logger.info("Starting Episode {}, Object {}, Trial {}, Step {}".format(
            _r.episode_idx, _r.garment_id, _r.trial_idx, self.step_idx)
        )

        # reset the robot to the home position
        exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))
        # create logger
        self._init_logger()
        # take point cloud
        logger.info("stage 3.1: capture pcd before action")
        obs, err = self._get_observation()
        # decision
        logger.info("stage 3.2: model inference")
        prediction_message, action_message, err = _r.running_inference.predict_action(
            obs,
            action_type=self._get_action_type(),
            vis=cfg.inference.args.vis_action,
        )
        self._latest_inference, self._latest_action = prediction_message, action_message
        if err is not None:
            return None, None, None, self._abort_with_error(err)
        self._log_after_prediction(prediction_message, action_message)
        # after decision
        if cfg.experiment.compat.debug:
            self._debug_visualize_points(action_message, obs)
        # execute decision
        self._log_before_action(obs)
        logger.info("stage 3.3: execute action")
        err = self._execute_action_failsafe(action_message)
        if err is not None:
            return None, None, None, self._abort_with_error(err)
        #  after action
        self._finalize_after_action(action_message, err)

        return obs, action_message, prediction_message, err

    def on_enter_crumpled(self):
        _r = ExperimentRegistry()
        logger.info("Starting Episode {}, Object {}, Trial {}, Step {}".format(
            _r.episode_idx, _r.garment_id, _r.trial_idx, self.step_idx)
        )
        self._action_loop()

        # protected section
        self.step_idx += 1
        self._latest_observation, self._latest_err = _r.exp.capture_pcd()
        self.update_condition()

    def on_enter_unreachable(self):
        _r = ExperimentRegistry()
        self._try_lift_to_center()

        # protected section
        self.update_condition()

    def on_enter_smoothed(self):
        _r = ExperimentRegistry()
        _, action_message, _, err = self._action_loop()
        while action_message.action_type in {ActionTypeDef.DRAG_HYBRID, ActionTypeDef.DRAG}:
            _, action_message, _, err = self._action_loop()

        # protected section
        if err is not None:
            self.current_state = GarmentStateDef.FAILED
            return
        self.step_idx += 1
        self._latest_observation, self._latest_err = _r.exp.capture_pcd()
        self.update_condition()

    def on_enter_folded_once(self):
        _r = ExperimentRegistry()
        _, action_message, _, err = self._action_loop()
        while action_message.action_type in {ActionTypeDef.DRAG_HYBRID, ActionTypeDef.DRAG}:
            _, action_message, _, err = self._action_loop()

        # protected section
        if err is not None:
            self.current_state = GarmentStateDef.FAILED
            return
        self.step_idx += 1
        self._latest_observation, self._latest_err = _r.exp.capture_pcd()
        self.update_condition()

    def on_enter_folded_twice(self):
        logger.info("stage 3.5: the end")
        # reset the robot to the home position
        _r = ExperimentRegistry()
        _r.exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))

    def update_condition(self):
        _r = ExperimentRegistry()
        payload = {
            "garment_operable": _r.exp.is_garment_on_table(self._latest_observation.mask_img),
            "garment_reachable": _r.exp.is_garment_reachable(self._latest_observation.mask_img),
            "garment_smooth_enough": True if _r.running_inference.predict_raw_action_type(self._latest_observation) != ActionTypeDef.FLING else False,
            "garment_need_drag": False,  # TODO: Implement this
            "garment_fling_drag_success": True,  # TODO: Implement this
            "garment_folded_once_success": self._latest_action is not None and self._latest_action.action_type == ActionTypeDef.FOLD_1,
            "garment_folded_twice_success": self._latest_action is not None and self._latest_action.action_type == ActionTypeDef.FOLD_2,
        }
        self.condition = GarmentMachineConditions(**payload)

    def dump(self, img_path: str):
        self._graph().write_png(img_path)


def _test():
    seed = time.time() * 1e6  # 1695502691057083.8, 1695503526499119.0
    logger.info(f"seed={seed}")
    random.seed(seed)

    total_success = 0
    total_failed = 0
    num_steps_arr = []

    GarmentStateMachine(disp=False).dump('statemachine_garment.png')
    for idx in range(100):
        m = GarmentStateMachine(disp=True)
        print(f">============== begin {idx + 1} trial ==============")
        while True:
            m.loop()
            if m.current_state.name == GarmentStateDef.SUCCESS:
                print("[result] =", m.current_state.name)
                break
            elif m.current_state.name == GarmentStateDef.FAILED:
                print("[result] =", m.current_state.name)
                break

        print(f">============== end {idx + 1} trial ==============")
        print("\n")
        if m.current_state.name == GarmentStateDef.SUCCESS:
            total_success += 1
            num_steps_arr.append(m.step_idx)
        else:
            total_failed += 1

    print(f"total success: {total_success}, total failed: {total_failed}")
    print(
        f"avg={sum(num_steps_arr) / (1e-8 + len(num_steps_arr))}, "
        f"max={max(num_steps_arr) if num_steps_arr else 'N/A'}, "
        f"min={min(num_steps_arr) if num_steps_arr else 'N/A'}"
    )


if __name__ == '__main__':
    _test()
