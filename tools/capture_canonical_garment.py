import os
import os.path as osp
import sys

import py_cli_interaction
from typing import Tuple, Optional, Iterable, Dict
import hydra
from omegaconf import DictConfig

sys.path.insert(0, osp.join("..", os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from planning.configs.config import config as planning_config
from manipulation.experiment_real import ExperimentReal
from common.experiment_base import convert_dict
from omegaconf import OmegaConf
# Experiment = None
from common.logging_utils import Logger as ExpLogger

def is_validate_garment_id(garment_id: str) -> bool:
    if garment_id == "":
        return False
    else:
        # TODO: verify garment id, check if it is in the database
        return True

def collect_real_data(cfg, exp: ExperimentReal):
    for obj_idx in range(100):
        logger.info("Input garment id...")
        garment_id = ""
        while not (is_validate_garment_id(garment_id) and continue_flag):
            garment_id = py_cli_interaction.must_parse_cli_string("input garment_id")
            continue_flag = py_cli_interaction.must_parse_cli_bool(
                "i have confirmed that the correct garment is selected and flattened"
            )

        # create logger
        experiment_logger = ExpLogger(
            namespace=cfg.logging.namespace, config=cfg.logging, tag=cfg.logging.tag
        )
        experiment_logger.init()
        experiment_logger.log_running_config(cfg)
        experiment_logger.log_commit(cfg.experiment.environment.project_root)
        experiment_logger.log_garment_id(garment_id)

        # take point cloud
        logger.info("stage 3.1: capture pcd!")

        obs, err = exp.capture_pcd()
        experiment_logger.log_pcd_raw("begin", obs.raw_virtual_pcd)
        experiment_logger.log_rgb("begin", obs.rgb_img)
        experiment_logger.log_mask("begin", obs.mask_img)

        experiment_logger.log_pcd_processed("begin", obs.valid_virtual_pcd)
        experiment_logger.close()


@hydra.main(
    config_path="config/real_experiment", config_name="experiment_real_tshirt_long", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    cfg.experiment.compat.use_real_robots = False
    cfg.experiment.planning = OmegaConf.create(convert_dict(planning_config))

    # create experiment
    exp = ExperimentReal(config=cfg.experiment)

    # start capturing garments
    collect_real_data(cfg, exp)


if __name__ == "__main__":
    main()
