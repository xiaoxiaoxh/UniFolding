import os
import os.path as osp
import sys

import py_cli_interaction
import hydra
from omegaconf import DictConfig
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.notification import get_bark_notifier
from common.statemachine import GarmentStateDef
from manipulation.statemachine_garment import GarmentStateMachine

from planning.configs.config import config_tshirt_long as planning_config_tshirt_long
from planning.configs.config import config_tshirt_short as planning_config_tshirt_short
from loguru import logger
from learning.inference_3d import Inference3D
from learning.datasets.runtime_dataset_real import RuntimeDataModuleReal
from learning.datasets.runtime_dataset_virtual import RuntimeDataModuleVirtual

from manipulation.experiment_real import ExperimentReal
from common.experiment_base import convert_dict

from common.registry import ExperimentRegistry
from common.train_util import train_model_with_hybrid_dataset, barrier
from tools.debug_controller import Client as DebugClient
from omegaconf import OmegaConf


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


def finalize_training(next_episode_idx: int, working_dir: str = None):
    import os
    logger.info(f"finalize_training called with {next_episode_idx}, {working_dir}")
    if working_dir is None:
        working_dir = os.getcwd()

    _home_dir = os.path.expanduser('~')
    target_dir = os.path.join(_home_dir, '.unifolding/config')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    with open(os.path.join(target_dir, 'EPISODE_IDX'), 'w') as f:
        f.write(str(next_episode_idx))

    with open(os.path.join(target_dir, 'WORKING_DIR'), 'w') as f:
        f.write(working_dir)

    return None


def collect_real_data():
    _r = ExperimentRegistry()
    cfg, exp = _r.cfg, _r.exp
    episode_idx: int = _r.episode_idx

    # create inference class
    inference = Inference3D(experiment=exp, **cfg.inference)
    logger.info(f'Starting Episode {episode_idx}!')
    _r.running_inference = inference

    fixed_garment_id = cfg.experiment.strategy.fixed_garment_id
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

        _r.garment_id = garment_id
        for trial_idx in range(cfg.experiment.strategy.trial_num_per_instance):
            _r.trial_idx = trial_idx
            logger.info(f"stage 2: inputs action type")
            m = GarmentStateMachine(disp=True)
            while True:
                m.loop()
                if m.current_state.name == GarmentStateDef.SUCCESS:
                    print("[result] =", m.current_state.name)
                    break
                elif m.current_state.name == GarmentStateDef.FAILED:
                    print("[result] =", m.current_state.name)
                    break

        _n = get_bark_notifier()
        err = _n.notify("[UniFolding] Time to change the cloth")
        if err is not None:
            logger.error(f'Failed to connect to notification server!')


@hydra.main(
    config_path="config/real_experiment", config_name="experiment_real_tshirt_long", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    global __DEBUG_CLIENT__
    reg = ExperimentRegistry()
    # hydra creates working directory automatically
    pred_output_dir = os.getcwd()
    logger.info(pred_output_dir)
    _n = get_bark_notifier()
    err = _n.notify("[UniFolding] Program Starts!!")
    if err is not None:
        logger.error(f'Failed to connect to notification server!')

    if cfg.inference.remote_debug.enable:
        logger.info(f"enable remote debug, url={cfg.inference.remote_debug.endpoint}")
        reg.debug_client = DebugClient(cfg.inference.remote_debug.endpoint)

    if cfg.experiment.compat.garment_type == 'tshirt_long':
        planning_config = planning_config_tshirt_long
    elif cfg.experiment.compat.garment_type == 'tshirt_short':
        planning_config = planning_config_tshirt_short
    else:
        raise NotImplementedError
    cfg.experiment.planning = OmegaConf.create(convert_dict(planning_config))
    # init
    runtime_output_dir = None
    episode_idx = cfg.experiment.strategy.start_episode
    logger.debug(f'start episode_idx: {episode_idx}')
    for episode_idx in range(cfg.experiment.strategy.start_episode,
                             cfg.experiment.strategy.start_episode + cfg.experiment.strategy.episode_num):
        if runtime_output_dir is not None:
            # load newest runtime checkpoint
            cfg.inference.model_path = runtime_output_dir
            cfg.inference.model_name = 'last'
        inference = Inference3D(**cfg.inference)
        # get init version of policy model
        policy_model = inference.model
        # get init state dict of policy model and model class for runtime training
        inference_model_state_dict = policy_model.state_dict()
        model_class = type(policy_model)

        if cfg.experiment.strategy.skip_data_collection_in_first_episode and \
                episode_idx == cfg.experiment.strategy.start_episode:
            pass
        else:
            try:
                # create experiment
                exp = ExperimentReal(config=cfg.experiment)
                exp.controller.actuator.open_gripper()
                # collect data
                logger.info(f"Begin to collect data for episode {episode_idx}!")
                reg.cfg = cfg
                reg.get_runtime_training_config = get_runtime_training_config
                reg.is_validate_garment_id = is_validate_garment_id
                reg.exp = exp
                reg.episode_idx = episode_idx
                collect_real_data()
            finally:
                exp.camera.stop()
                reg.exp = None
                del exp

        if cfg.experiment.strategy.barrier.enable:
            # use barrier to synchronize with the virtual data collection process in Stage 3
            logger.info(f'Waiting for barrier...')
            barrier(cfg.experiment.strategy.barrier.tag, cfg.experiment.strategy.barrier.num_processes)
            logger.info(f'Barrier passed!')

        # create runtime datamodule
        if cfg.experiment.strategy.use_online_dataset:
            start_episode_idx = max(0, episode_idx - cfg.experiment.strategy.max_memory_size + 1)
            # only use data from the last few episodes
            cfg.experiment.runtime_training_config_override.runtime_datamodule.episode_range = \
                (start_episode_idx, episode_idx + 1)
        runtime_training_config = get_runtime_training_config(cfg.inference.model_path,
                                                              cfg.experiment.runtime_training_config_override)
        # create static datamodule (virtual dataset)
        virtual_datamodule = RuntimeDataModuleVirtual(logging_dir=cfg.logging.path,
                                                      namespace='virtual',
                                                      **runtime_training_config.virtual_datamodule)
        virtual_datamodule.prepare_data()
        # create runtime datamodule (real dataset)
        runtime_datamodule = RuntimeDataModuleReal(logging_dir=cfg.logging.path,
                                                   namespace=cfg.logging.namespace,
                                                   tag=cfg.logging.tag,
                                                   **runtime_training_config.runtime_datamodule)
        runtime_datamodule.prepare_data()
        runtime_dataset_size = len(runtime_datamodule.train_dataset)
        if runtime_dataset_size >= cfg.experiment.strategy.warmup_sample_num:
            # create runtime model
            runtime_model = model_class(**runtime_training_config.model)
            runtime_model.load_state_dict(inference_model_state_dict, strict=False)

            # create runtime output directory
            runtime_output_dir = osp.join(pred_output_dir, 'episode{:03d}'.format(episode_idx))
            os.makedirs(runtime_output_dir, exist_ok=True)
            runtime_training_config.logger.run_name = cfg.logging.note + '_real-episode{:03d}'.format(
                episode_idx)
            # runtime training
            train_model_with_hybrid_dataset(runtime_output_dir, runtime_training_config,
                                            [virtual_datamodule, runtime_datamodule], runtime_model)
    if cfg.experiment.strategy.finalize_training:
        finalize_training(episode_idx + 1, runtime_output_dir)


if __name__ == "__main__":
    main()
