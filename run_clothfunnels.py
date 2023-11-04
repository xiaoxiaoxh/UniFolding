import os
import os.path as osp
import sys

import py_cli_interaction
from typing import Optional
import numpy as np
import hydra
from omegaconf import DictConfig
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from planning.configs.config import config_tshirt_long as planning_config_tshirt_long
from planning.configs.config import config_tshirt_short as planning_config_tshirt_short
from planning.error_config import error_code
import matplotlib.pyplot as plt
import time
from loguru import logger
from learning.inference_clothfunnels import InferenceClothFunnels

from manipulation.experiment_real import ExperimentReal
from common.experiment_base import convert_dict

# Experiment = None
from common.logging_utils import Logger as ExpLogger
from common.visualization_util import visualize_pc_and_grasp_points
from common.datamodels import ActionTypeDef, ActionMessage
from tools.debug_controller import Client as DebugClient
from omegaconf import OmegaConf

import requests

__NOTIFICATION_KEY__ = os.environ.get("CONFIG_NOTIFICATION_KEY")
if __NOTIFICATION_KEY__ != "" and __NOTIFICATION_KEY__ is not None:
    CONFIG_NOTIFICATION_URL = "https://api.day.app/" + __NOTIFICATION_KEY__ + "/{}?isArchive=1"
    __NOTIFICATION_SESSION__ = requests.session()
else:
    CONFIG_NOTIFICATION_URL = None
    __NOTIFICATION_SESSION__ = None

__ACTION_TYPE_OVERRIDE_SET__ = [
    None,
    ActionTypeDef.to_string(ActionTypeDef.FLING),
    ActionTypeDef.to_string(ActionTypeDef.PICK_AND_PLACE),
    ActionTypeDef.to_string(ActionTypeDef.FOLD_1),
    ActionTypeDef.to_string(ActionTypeDef.FOLD_2),
]
__DEBUG_CLIENT__: Optional[DebugClient] = None

def is_validate_garment_id(garment_id: str) -> bool:
    if garment_id == "":
        return False
    else:
        # TODO: verify garment id, check if it is in the database
        return True


def get_runtime_training_config(model_path: str, runtime_model_config_override: DictConfig) -> DictConfig:
    """log original model configuration and override with runtime configuration

    Args:
        model_path (str): path to model checkpoint directory
        runtime_model_config_override (DictConfig): runtime configuration override
    """
    with open(osp.join(model_path, 'config.yaml'), 'r') as f:
        model_config = yaml.load(f, Loader=yaml.SafeLoader)['config']
    model_config = DictConfig(model_config)
    model_config.merge_with(runtime_model_config_override)
    # copy and merge datamodule config
    for key, value in model_config.datamodule.items():
        if key not in model_config.runtime_datamodule:
            model_config.runtime_datamodule[key] = value
    # copy and merge virtual datamodule config
    for key, value in model_config.datamodule.items():
        if key not in model_config.virtual_datamodule:
            model_config.virtual_datamodule[key] = value
    return model_config


def confirm_abort(exp: ExperimentReal):
    _confirm = py_cli_interaction.must_parse_cli_bool("abort experiment?")
    if _confirm:
        logger.error("aborting experiment!")
        exp.camera.stop()
        raise Exception("abort experiment")


def get_remote_action_type_str(client: DebugClient=None) -> str:
    if client is None:
        logger.debug("client is None, remote ActionType=None")
        return 'null'
    else:
        try:
            res = client.get_action_type()
        except Exception as e:
            logger.warning(e)
            res = None
        logger.debug("remote ActionType={res}")
        return ActionTypeDef.to_string(res)


def collect_real_data(cfg, exp: ExperimentReal, episode_idx: int = 0, fixed_garment_id: str = None):
    # create inference class
    inference = InferenceClothFunnels(experiment=exp, **cfg.inference)
    counter = {
        'step_num': 0,
    }
    logger.info(f'Starting Episode {episode_idx}!')

    for obj_idx in range(cfg.experiment.strategy.instance_num_per_episode):
        logger.info("stage 1: inputs garment id")
        if fixed_garment_id is None:
            garment_id = ""
            while not (is_validate_garment_id(garment_id) and continue_flag):
                garment_id = py_cli_interaction.must_parse_cli_string("input garment_id")
                continue_flag = py_cli_interaction.must_parse_cli_bool(
                    "i have confirmed that the correct garment is selected and mounted"
                )
        else:
            garment_id = fixed_garment_id
        for trial_idx in range(cfg.experiment.strategy.trial_num_per_instance):
            logger.info("Starting Episode {}, Object {}, Trial {}".format(episode_idx, obj_idx, trial_idx))
            # reset the robot to the home position
            if cfg.experiment.compat.use_real_robots:
                exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))
            logger.info("stage 2: randomize the garment")

            last_action_type = None
            if cfg.experiment.strategy.random_lift_in_each_trial and cfg.experiment.compat.use_real_robots:
                obs, err = exp.capture_pcd()
                # obs.valid_virtual_pts  # (N, 3)
                joint_values_message = exp.choose_random_pt(obs.valid_virtual_pts)
                logger.info("Implement LIFT action")
                exp.execute_action(action=joint_values_message)
            for step_idx in range(cfg.experiment.strategy.step_num_per_trial):
                logger.info("Starting Episode {}, Object {}, Trial {}, Step {}".format(
                    episode_idx, obj_idx, trial_idx, step_idx))
                # reset the robot to the home position
                if cfg.experiment.compat.use_real_robots:
                    exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))
                # create logger
                experiment_logger = ExpLogger(
                    namespace=cfg.logging.namespace, config=cfg.logging, tag=cfg.logging.tag
                )
                experiment_logger.init()
                experiment_logger.log_running_config(cfg)
                experiment_logger.log_commit(cfg.experiment.environment.project_root)
                experiment_logger.log_calibration(
                    exp.transforms.camera_to_world_transform,
                    exp.transforms.left_robot_to_world_transform,
                    exp.transforms.right_robot_to_world_transform,
                )
                experiment_logger.log_garment_id(garment_id)                
                experiment_logger.log_episode_idx(episode_idx)
                experiment_logger.log_trial_idx(trial_idx)
                experiment_logger.log_action_step(step_idx)

                # take point cloud
                logger.info("stage 3.1: capture pcd before action")

                if cfg.experiment.strategy.check_grasp_failure_before_action:
                    grasp_failure = True
                    while grasp_failure:
                        obs, err = exp.capture_pcd()
                        grasp_failure = not exp.is_garment_reachable(obs.mask_img)
                        if grasp_failure and cfg.experiment.compat.use_real_robots:
                            joint_values_message = exp.choose_random_pt(obs.valid_virtual_pts)
                            logger.info("Grasp failure detected! Implement LIFT action!")
                            exp.execute_action(action=joint_values_message)
                else:
                    obs, err = exp.capture_pcd()

                if cfg.inference.args.vis_action:
                    save_path = '/home/xuehan/Desktop/CoRL_vis/ClothFunnels'
                    plt.figure()
                    plt.axis('off')
                    plt.imshow(obs.projected_rgb_img)
                    plt.title('Input RGB image', fontsize=25)
                    plt.savefig(os.path.join(save_path, f'{time.strftime("%Y-%m-%d %H-%M-%S")+" "+str(time.time())}.png'))
                    plt.show()
                experiment_logger.log_pcd_raw("begin", obs.raw_virtual_pcd, only_npz=True)
                experiment_logger.log_rgb("begin", obs.rgb_img)
                experiment_logger.log_mask("begin", obs.mask_img)

                experiment_logger.log_pcd_processed("begin", obs.valid_virtual_pcd, only_npz=True)

                logger.info("stage 3.2: model inference")
                if cfg.inference.args.action_type_override.enable:
                    raise NotImplementedError
                else:
                    # predict by the classifier and the state machine
                    action_type = inference.predict_raw_action_type(obs)

                if last_action_type == ActionTypeDef.FOLD_1:
                    # change action_type to fold2 directly
                    action_type = ActionTypeDef.FOLD_2
                elif last_action_type == ActionTypeDef.FOLD_2:
                    action_type = ActionTypeDef.DONE

                # predict action
                prediction_message, action_message, err = inference.predict_action(obs, action_type)

                # record last action type
                last_action_type = action_type

                # handle errors
                if err is not None:
                    logger.warning(f'{err}')
                    if action_message.action_type == ActionTypeDef.FAIL:
                        counter['step_num'] += 1
                        experiment_logger.finalize()
                        experiment_logger.close()
                        break
                    exp.controller.actuator.open_gripper()
                    if not cfg.experiment.strategy.skip_all_errors:
                        raise err

                # after decision
                experiment_logger.log_action_type(
                    ActionTypeDef.to_string(action_message.action_type)
                )

                if action_message.action_type != ActionTypeDef.DONE:
                    left_pick_point_in_virtual, right_pick_point_in_virtual \
                        = exp.get_pick_points_in_virtual(action_message)
                    left_place_point_in_virtual, right_place_point_in_virtual \
                        = exp.get_place_points_in_virtual(action_message)
                    if cfg.experiment.compat.debug:
                        visualize_pc_and_grasp_points(
                            obs.raw_virtual_pts,
                            left_pick_point=left_pick_point_in_virtual[:3],
                            right_pick_point=right_pick_point_in_virtual[:3],
                            left_place_point=left_place_point_in_virtual[:3],
                            right_place_point=right_place_point_in_virtual[:3],
                            pc_colors=np.asarray(obs.raw_virtual_pcd.colors)
                        )

                # execute decision
                logger.info("stage 3.3: execute action")
                if action_message.action_type not in (ActionTypeDef.DONE, ActionTypeDef.FAIL):
                    if cfg.experiment.compat.use_real_robots:
                        err = exp.execute_action(action_message)
                elif action_message.action_type == ActionTypeDef.DONE:
                    logger.warning(f"Task done! Skipping action now...")
                elif action_message.action_type == ActionTypeDef.FAIL:
                    if cfg.experiment.strategy.skip_all_errors:
                        logger.warning(str(err))

                if err is not None:
                    # execution failed
                    exp.controller.actuator.open_gripper()
                    exp.controller.move_home_with_plan()
                    if not cfg.experiment.strategy.skip_all_errors:
                        raise err

                #  after action
                if not cfg.experiment.compat.only_capture_pcd_before_action:
                    logger.info("stage 3.4: capture pcd after action")
                    if cfg.experiment.compat.use_real_robots:
                        exp.execute_action(action=ActionMessage(action_type=ActionTypeDef.HOME))
                    obs, _ = exp.capture_pcd()
                    experiment_logger.log_pcd_raw("end", obs.raw_virtual_pcd, only_npz=True)
                    experiment_logger.log_rgb("end", obs.rgb_img)
                    experiment_logger.log_mask("end", obs.mask_img)
                    experiment_logger.log_pcd_processed("end", obs.valid_virtual_pcd, only_npz=True)

                counter['step_num'] += 1
                try:
                    experiment_logger.finalize()
                    experiment_logger.close()
                except yaml.representer.RepresenterError as e:
                    logger.error(e)
                    logger.debug(experiment_logger._metadata)

                if action_message.action_type in (ActionTypeDef.DONE, ActionTypeDef.FAIL) or \
                    (err is not None and err != error_code.grasp_failed):
                    # early stopping for this trial
                    logger.warning('Early stopping for this trial!')
                    break

            if cfg.experiment.strategy.demo_mode:
                # stop all trials for this garment (for demo only)
                break
            
        if __NOTIFICATION_SESSION__ is not None:
            try:
                __NOTIFICATION_SESSION__.post(CONFIG_NOTIFICATION_URL.format("[UniFolding] Time to change the cloth"))
            except Exception as e:
                logger.error(f'Failed to connect to notification server!')


@hydra.main(
    config_path="config/real_experiment", config_name="experiment_clothfunnels_tshirt_long", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    global __DEBUG_CLIENT__
    # hydra creates working directory automatically
    pred_output_dir = os.getcwd()
    logger.info(pred_output_dir)
    if __NOTIFICATION_SESSION__ is not None:
        __NOTIFICATION_SESSION__.post(CONFIG_NOTIFICATION_URL.format("[UniFolding] Program Starts!!"))
    
    if cfg.inference.remote_debug.enable:
        logger.info(f"enable remote debug, url={cfg.inference.remote_debug.endpoint}")
        __DEBUG_CLIENT__ = DebugClient(cfg.inference.remote_debug.endpoint)

    if cfg.experiment.compat.garment_type == 'tshirt_long':
        planning_config = planning_config_tshirt_long
    elif cfg.experiment.compat.garment_type == 'tshirt_short':
        planning_config = planning_config_tshirt_short
    else:
        raise NotImplementedError
    cfg.experiment.planning = OmegaConf.create(convert_dict(planning_config))
    # init
    episode_idx = cfg.experiment.strategy.start_episode
    logger.debug(f'start episode_idx: {episode_idx}')
    for episode_idx in range(cfg.experiment.strategy.start_episode,
                             cfg.experiment.strategy.start_episode + cfg.experiment.strategy.episode_num):
        try:
            # create experiment
            exp = ExperimentReal(config=cfg.experiment)
            if cfg.experiment.compat.use_real_robots:
                exp.controller.actuator.open_gripper()
            # collect data
            logger.info(f"Begin to collect data for episode {episode_idx}!")
            collect_real_data(cfg, exp, episode_idx)
        finally:
            exp.camera.stop()
            del exp


if __name__ == "__main__":

    main()
