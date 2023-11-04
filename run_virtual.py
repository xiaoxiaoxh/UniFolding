import sys
import os
import os.path as osp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
import yaml
import torch.multiprocessing as mp
import copy
import psutil
import numpy as np
from loguru import logger
from omegaconf import DictConfig
from typing import Optional
from manipulation.experiment_virtual import ExperimentVirtual
from learning.inference_3d import Inference3D
from learning.components.obj_data_loader import CLOTH3DObjLoader
from learning.datasets.runtime_dataset_virtual import RuntimeDataModuleVirtual
from learning.datasets.vr_dataset import VirtualRealityDataModule
from common.train_util import train_model_with_hybrid_dataset, barrier
from common.datamodels import ActionTypeDef, AnnotationFlag
from common.logging_utils import Logger as ExpLogger
from common.visualization_util import visualize_pc_and_grasp_points

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
    return model_config

def collect_virtual_data(rank: int, pid: int, cfg: DictConfig,
                         obj_loader: CLOTH3DObjLoader, episode_idx: int,
                         exp: Optional[ExperimentVirtual] = None) -> None:
    pps = psutil.Process(pid=pid)
    cfg = copy.deepcopy(cfg)
    cfg.experiment.environment.seed = cfg.experiment.environment.seed + \
                                      episode_idx * cfg.experiment.strategy.num_processes + rank
    # create experiment
    if exp is None:
        exp = ExperimentVirtual(cfg.experiment)
    # set category metadata
    exp.set_category_meta(obj_loader.category_meta_dict)
    # create inference
    inference = Inference3D(experiment=exp, **cfg.inference)
    counter = {
        'step_num' : 0,
    }
    if rank == 0:
        logger.info(f'Starting Episode {episode_idx}!')
    for obj_idx, obj_path in enumerate(obj_loader.split_obj_path_list[rank]):
        if obj_idx >= cfg.experiment.strategy.instance_num_per_episode_per_proc:
            break
        garment_id = os.path.dirname(obj_path).split('/')[-1]
        for trial_idx in range(cfg.experiment.strategy.trial_num_per_instance):
            if rank == 0:
                logger.info(f'Loading object {obj_path}')
            exp.load_obj(obj_path)
            # reset garment state randomly
            is_random_fold_init_state = False
            if np.random.random() < cfg.experiment.strategy.random_fold_config.random_fold_data_ratio:
                # randomly decide whether to randomly fold the garment or not
                logger.info(f'Episode {episode_idx}, Garment idx {obj_idx}, randomly fold the garment...')
                exp.random_fold()
                is_random_fold_init_state = True
            else:
                logger.info(f'Episode {episode_idx}, Garment idx {obj_idx}, randomly grasp the garment')
                exp.random_grasp_and_drop()
            for step_idx in range(cfg.experiment.strategy.step_num_per_trial):
                # check if parent process is alive
                if pps.status() in (psutil.STATUS_DEAD, psutil.STATUS_STOPPED):
                    logger.error('Parent Process {} has stopped, rank {} quit now!!'.format(pid, rank))
                    os._exit(0)

                # init logger
                experiment_logger = ExpLogger(namespace='virtual',
                                              config=cfg.logging,
                                              tag=cfg.logging.tag)
                experiment_logger.init()
                # log the whole config
                experiment_logger.log_running_config(cfg)
                experiment_logger.log_model(cfg.inference.model_path, cfg.inference.model_name)
                experiment_logger.log_garment_id(garment_id)
                experiment_logger.log_action_step(step_idx)
                experiment_logger.log_episode_idx(episode_idx)
                experiment_logger.log_trial_idx(trial_idx)

                # capture point cloud (before action)
                if rank == 0:
                    logger.info(
                        f'Episode {episode_idx}, Garment idx {obj_idx}, Trial {trial_idx} Step {step_idx}, '
                        f'before action, capturing point cloud from camera...')
                obs_message, exception_message = exp.capture_pcd()
                experiment_logger.log_pcd_processed("begin", obs_message.valid_virtual_pcd, only_npz=True)
                experiment_logger.log_particles("begin", obs_message.particle_xyz)
                experiment_logger.log_nocs("begin", obs_message.valid_nocs_pts)
                # calculate reward (before action)
                reward_dict = exp.get_deformable_reward(obs_message)
                experiment_logger.log_reward("begin", reward_dict)

                # inference
                pred, action, _ = inference.predict_action(obs_message,
                                                           action_type=ActionTypeDef.from_string(cfg.inference.args.action_type_override.type)
                                                           if cfg.inference.args.action_type_override.enable else None,
                                                           vis=cfg.inference.args.vis_action,
                                                           running_seed=None)
                is_best_action = False
                if is_random_fold_init_state and step_idx == 0:  # only override action for the first step
                    # randomly choose whether to use the best action or not
                    if np.random.random() < cfg.experiment.strategy.random_fold_config.best_action_ratio:
                        # override action to be the best action for random fold init state
                        logger.info(f'Episode {episode_idx}, Garment idx {obj_idx}, Step {step_idx}, randomly choose best action...')
                        action = exp.get_best_fling_action()
                        is_best_action = True
                left_pick_point_in_virtual, right_pick_point_in_virtual = exp.get_pick_points_in_virtual(action)
                if cfg.experiment.compat.debug:
                    visualize_pc_and_grasp_points(obs_message.valid_virtual_pts,
                                                  left_pick_point_in_virtual[:3],
                                                  right_pick_point_in_virtual[:3],
                                                  pred.grasp_point_all)
                experiment_logger.log_pose_prediction_virtual("begin", pred.grasp_point_all)
                experiment_logger.log_pose_gripper_virtual("begin", left_pick_point_in_virtual, right_pick_point_in_virtual,
                                                           is_best=is_best_action)
                experiment_logger.log_action_type(ActionTypeDef.to_string(action.action_type))

                # execute action
                exe_result = exp.execute_action(action)

                # capture point cloud (after action)
                if rank == 0:
                    logger.info(
                        f'Episode {episode_idx}, Garment idx {obj_idx}, Step {step_idx}, after action, capturing point cloud from camera...')
                obs_message, exception_message = exp.capture_pcd()
                experiment_logger.log_particles("end", obs_message.particle_xyz)
                # calculate reward (after action)
                reward_dict = exp.get_deformable_reward(obs_message)
                experiment_logger.log_reward("end", reward_dict)

                counter['step_num'] += 1
                # for virtual data only
                experiment_logger.log_processed_file(str(AnnotationFlag.COMPLETED.value))
                # for virtual data only
                experiment_logger.log_empty_annotation_file()
                experiment_logger.close()
# %%
# main script
@hydra.main(config_path="config/virtual_experiment_stage2", config_name="experiment_virtual_tshirt_long",
            version_base='1.1')
def main(cfg: DictConfig) -> None:
    # hydra creates working directory automatically
    pred_output_dir = os.getcwd()
    logger.info(pred_output_dir)

    assert cfg.experiment.strategy.num_processes == cfg.experiment.obj_loader.num_splits, \
        'num_processes must be equal to num_splits'
    # create obj loader
    obj_loader = CLOTH3DObjLoader(**cfg.experiment.obj_loader)
    # init
    runtime_output_dir = None
    if cfg.experiment.strategy.num_processes == 1:
        # create one single experiment
        experiment = ExperimentVirtual(cfg.experiment)
    else:
        # create experiments inside each process
        experiment = None

    for episode_idx in range(cfg.experiment.strategy.start_episode, cfg.experiment.strategy.start_episode +
                                                                    cfg.experiment.strategy.episode_num):
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

        # shuffle instances
        if cfg.experiment.strategy.shuffle_instances:
            obj_loader.shuffle_instances()

        if (cfg.experiment.strategy.skip_data_collection_in_first_episode and \
                episode_idx == cfg.experiment.strategy.start_episode) or \
                cfg.experiment.strategy.skip_all_data_collection:
            pass
        else:
            pid = os.getpid()
            logger.info(f'Create {cfg.experiment.strategy.num_processes} processes for data collection...')
            if cfg.experiment.strategy.num_processes == 1:
                # directly call data collection function
                collect_virtual_data(0, pid, cfg, obj_loader, episode_idx, experiment)
            else:
                # create processes for parallel data collection
                mp.spawn(collect_virtual_data,
                         args=(pid, cfg, obj_loader, episode_idx),
                         nprocs=cfg.experiment.strategy.num_processes,
                         join=True,
                         daemon=True)

        if cfg.experiment.strategy.barrier.enable:
            # use barrier to synchronize with the real data collection process in Stage 3
            logger.debug(f'Waiting for barrier...')
            barrier(cfg.experiment.strategy.barrier.tag, cfg.experiment.strategy.barrier.num_processes)
            logger.debug(f'Barrier passed!')

        if cfg.experiment.strategy.skip_all_model_training:
            # skip all model training, only perform data collection
            continue

        # create runtime datamodule
        if cfg.experiment.strategy.use_online_dataset:
            start_episode_idx = max(0, episode_idx - cfg.experiment.strategy.max_memory_size + 1)
            # only use data from the last few episodes
            cfg.experiment.runtime_training_config_override.runtime_datamodule.episode_range = \
                (start_episode_idx, episode_idx+1)
        runtime_training_config = get_runtime_training_config(cfg.inference.model_path,
                                                              cfg.experiment.runtime_training_config_override)
        # create static datamodule (VR dataset)
        static_datamodule = VirtualRealityDataModule(**runtime_training_config.datamodule)
        static_datamodule.prepare_data()
        # create runtime datamodule (virtual dataset)
        runtime_datamodule = RuntimeDataModuleVirtual(logging_dir=cfg.logging.path,
                                                      namespace='virtual',
                                                      tag=cfg.logging.tag,
                                                      **runtime_training_config.runtime_datamodule)
        runtime_datamodule.prepare_data()
        runtime_dataset_size = len(runtime_datamodule.train_dataset)
        if runtime_dataset_size >= cfg.experiment.strategy.warmup_sample_num:
            # create runtime model
            runtime_model = model_class(**runtime_training_config.model)
            try:
                runtime_model.load_state_dict(inference_model_state_dict, strict=False)
            except RuntimeError as e:
                logger.warning(f'Failed to load state dict of model! Only load filtered state dict...')
                filtered_model_state_dict = {}
                for key, item in inference_model_state_dict.items():
                    if 'virtual_reward_head' not in key:
                        filtered_model_state_dict[key] = item
                runtime_model.load_state_dict(filtered_model_state_dict, strict=False)

            # create runtime output directory
            runtime_output_dir = osp.join(pred_output_dir, 'episode{:03d}'.format(episode_idx))
            os.makedirs(runtime_output_dir, exist_ok=True)
            runtime_training_config.logger.run_name = cfg.logging.note + '_virtual-episode{:03d}'.format(episode_idx)
            # runtime training
            train_model_with_hybrid_dataset(runtime_output_dir, runtime_training_config,
                                            [static_datamodule, runtime_datamodule], runtime_model)



if __name__ == '__main__':
    main()
