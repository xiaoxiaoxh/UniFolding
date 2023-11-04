import os
import time
import yaml
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional, Any, Callable
import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.loggers import MLFlowLogger
import requests
from loguru import logger


class SlackMessenger:
    def __init__(
            self,
            identifier: str,
            webhook_url: str = None,
            msg_fn_on_start: Optional[Callable[[Any], str]] = None,
            msg_fn_on_epoch_end: Optional[Callable[[Any], str]] = None,
            msg_fn_on_finish: Optional[Callable[[Any], str]] = None,
            msg_fn_on_failure: Optional[Callable[[Any], str]] = None,
    ):
        self.identifier = identifier
        if webhook_url is None:
            self.webhook_url = os.environ.get('SLACK_WEBHOOK_URL', None)
        else:
            self.webhook_url = webhook_url
        self.session = requests.Session()

        self.msg_fn_on_start = (lambda ctx: "Experiment started!") if msg_fn_on_start is None else msg_fn_on_start
        self.msg_fn_on_epoch_end = (
            lambda ctx: "Epoch finished!"
        ) if msg_fn_on_epoch_end is None else msg_fn_on_epoch_end
        self.msg_fn_on_finish = (lambda ctx: "Experiment finished!") if msg_fn_on_finish is None else msg_fn_on_finish
        self.msg_fn_on_failure = (lambda ctx: "Experiment failed!") if msg_fn_on_failure is None else msg_fn_on_failure

    def _send(self, text: str) -> Optional[Exception]:
        if self.webhook_url is None:
            logger.debug(f"No webhook url provided!")
            return Exception("No webhook url provided!")
        else:
            resp = requests.post(self.webhook_url, json={'text': text})
            if resp.status_code != 200:
                logger.debug(f"Failed to send message to slack! Status code: {resp.status_code}")
                return Exception(f"Failed to send message to slack! Status code: {resp.status_code}")
            else:
                logger.debug(f"Successfully sent message to slack! Status code: {resp.status_code}")
                return None

    def on_start(self, ctx: Any = None) -> Optional[Exception]:
        text = f"[{self.identifier}] {self.msg_fn_on_start(ctx)}"
        return self._send(text)

    def on_epoch_end(self, ctx: Any = None) -> Optional[Exception]:
        text = f"[{self.identifier}] {self.msg_fn_on_epoch_end(ctx)}"
        return self._send(text)

    def on_finish(self, ctx: Any = None) -> Optional[Exception]:
        text = f"[{self.identifier}] {self.msg_fn_on_finish(ctx)}"
        return self._send(text)

    def on_failure(self, ctx: Any = None) -> Optional[Exception]:
        text = f"[{self.identifier}] {self.msg_fn_on_failure(ctx)}"
        return self._send(text)


def train_model_with_hybrid_dataset(output_dir: str, cfg: DictConfig, datamodule_list: List[pl.LightningDataModule],
                                    model: pl.LightningModule) -> None:
    print(f"Working directory: {output_dir}")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    model.batch_size = cfg.datamodule.batch_size
    logger = MLFlowLogger(**cfg.logger)
    all_config = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'output_dir': output_dir,
        # 'wandb': wandb_meta
    }
    yaml.dump(all_config, open(os.path.join(output_dir, 'config.yaml'), 'w'), default_flow_style=False)
    logger.log_hyperparams(all_config)

    assert pl.__version__ >= '2.0.2'
    if 'gpus' in cfg.trainer:
        # for campatibility with older version of pytorch lightning
        del cfg.trainer['gpus']
    if 'resume_from_checkpoint' in cfg.trainer:
        # for campatibility with older version of pytorch lightning
        del cfg.trainer['resume_from_checkpoint']
    assert len(datamodule_list) <= 2, 'Only support one or two datasets for now!'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="{epoch}-{val_loss/dataloader_idx_1:.4f}" if len(datamodule_list) == 2 else "{epoch}-{val_loss:.4f}",
        # TODO: more flexible monitor
        monitor='val_loss/dataloader_idx_1' if len(datamodule_list) == 2 else 'val_loss',
        save_last=True,
        save_top_k=1,
        mode='min',
        save_weights_only=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        default_root_dir=output_dir,
        enable_checkpointing=True,
        logger=logger,
        check_val_every_n_epoch=1,
        **cfg.trainer)
    if len(datamodule_list) == 1:
        trainer.fit(model=model, datamodule=datamodule_list[0])
    else:
        train_dataloaders = [dm.train_dataloader() for dm in datamodule_list]
        combined_train_loader = CombinedLoader(train_dataloaders, mode="min_size")
        val_dataloaders = [dm.val_dataloader() for dm in datamodule_list]
        trainer.fit(model=model, train_dataloaders=combined_train_loader, val_dataloaders=val_dataloaders)

    # log artifacts
    logger.experiment.log_artifact(logger.run_id, output_dir)
    print(f"Log artifacts to {logger.run_id}!")




class FDBarrier:
    def __init__(self, path) -> None:
        self.path = path
        os.mkfifo(path)

    def __del__(self):
        os.remove(self.path)


def reset_barrier(tag: str = None, domain: str = "/tmp/unifolding") -> None:
    if tag is None:
        for f in os.listdir(domain):
            os.remove(os.path.join(domain, f))
    else:
        barrier_master = os.path.join(domain, f"{tag}.master")
        if os.path.exists(barrier_master):
            os.remove(barrier_master)


def barrier(tag: str, n: int, domain: str = "/tmp/unifolding") -> None:
    """
    a barrier that blocks until n processes reach this point
    """
    is_master = False
    _b = None  # lifetime is the same as the function

    # create the domain if not exists
    if not os.path.exists(domain):
        os.mkdir(domain)

    # check if the master exists
    barrier_fd_path = os.path.join(domain, f"{tag}.master")
    if not os.path.exists(barrier_fd_path):
        # become master
        try:
            _b = FDBarrier(os.path.join(domain, f"{tag}.master"))
            is_master = True
        except FileExistsError:
            pass
        except Exception as e:
            raise e

    if is_master:
        # become master
        with open(barrier_fd_path, "r") as pipe:
            n_lines = 0
            while n_lines < n - 1:
                time.sleep(0.1)
                res = pipe.readlines()
                n_lines += len(res)
    else:
        with open(barrier_fd_path, 'w') as pipe:
            pipe.write(f'{os.getpid()}\n')
        while True:
            if not os.path.exists(barrier_fd_path):
                break
            else:
                time.sleep(0.1)