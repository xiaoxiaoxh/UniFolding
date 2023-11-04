import sys
import os
import os.path as osp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from learning.datasets.vr_dataset import VirtualRealityDataModule
from learning.net.ufo_net import UFONet

# %%
# main script
@hydra.main(config_path="config/virtual_experiment_stage1", config_name="train_virtual_tshirt_long", version_base='1.1')
def main(cfg: DictConfig) -> None:
    # hydra creates working directory automatically
    print(os.getcwd())
    os.mkdir("checkpoints")

    datamodule = VirtualRealityDataModule(**cfg.datamodule)
    model = UFONet(**cfg.model)
    model.batch_size = cfg.datamodule.batch_size

    # category = os.path.dirname(cfg.datamodule.h5_path)
    # cfg.logger.tags.append(category)
    # logger = pl.loggers.WandbLogger(
    #     project=os.path.basename(__file__),
    #     **cfg.logger)
    # wandb_run = logger.experiment
    # wandb_meta = {
    #     'run_name': wandb_run.name,
    #     'run_id': wandb_run.id
    # }

    # logger = pl.loggers.TensorBoardLogger("tb_logs", **cfg.logger)
    logger = MLFlowLogger(**cfg.logger)

    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'output_dir': os.getcwd(),
        # 'wandb': wandb_meta
    }
    yaml.dump(all_config, open('config.yaml', 'w'), default_flow_style=False)
    logger.log_hyperparams(all_config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.4f}",
        monitor='val_loss',
        save_last=True,
        save_top_k=1,
        mode='min', 
        save_weights_only=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        default_root_dir=os.getcwd(),
        enable_checkpointing=True,
        logger=logger,
        check_val_every_n_epoch=1,
        **cfg.trainer)
    trainer.fit(model=model, datamodule=datamodule)

    # log artifacts
    logger.experiment.log_artifact(logger.run_id, os.getcwd())

# %%
# driver
if __name__ == "__main__":
    main()
