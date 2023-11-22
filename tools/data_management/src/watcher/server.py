import dataclasses
import io
import json
import os
import os.path as osp
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Optional, List, Tuple

import minio
import requests
import yaml
from loguru import logger

from src.common.config import WatcherConfig
from src.common.datamodels import LogPoint
from src.client.client import Client
from src.client.api.default import root_get, upload_logs_v1_logs_upload_post
from src.common.utils import DelayedKeyboardInterrupt

_CONFIG_ARCHIVE_PATH = "archives"


@dataclasses.dataclass
class WatcherContext:
    opt: WatcherConfig
    api_client: Client
    oss_client: Optional[minio.Minio]


@logger.catch
def prepare_run(opt: WatcherConfig) -> Tuple[Optional[WatcherContext], Optional[Exception]]:
    if not osp.exists(opt.app_watchPath):
        logger.error(f"watchPath {opt.app_watchPath} does not exist")
        return None, Exception(f"watchPath {opt.app_watchPath} does not exist")

    if not osp.exists(archive_path := osp.join(opt.app_watchPath, _CONFIG_ARCHIVE_PATH)):
        os.makedirs(archive_path, exist_ok=True)
        logger.info(f"create archive path: {archive_path}")

    oss_client = None
    # >>>>>>>>> commented out to disable OSS <<<<<<<<<
    oss_client = minio.Minio(endpoint=f"{opt.oss_host}:{opt.oss_port}",
                             access_key=opt.oss_accessKey,
                             secret_key=opt.oss_secretKey,
                             secure=opt.oss_tls)
    try:
        oss_client.list_buckets()
        found = oss_client.bucket_exists(opt.oss_bucket)
        if not found:
            return None, Exception(f"bucket '{opt.oss_bucket}' does not exist")
        logger.info(f"oss connection established")
    except Exception as e:
        logger.error(f"failed to connect to oss: {e}")
        return None, e
    # >>>>>>>>> Commented out to Disable OSS <<<<<<<<<

    api_client = Client(base_url=opt.api_url, verify_ssl="https" in opt.api_url)

    try:
        root_get.sync_detailed(client=api_client)
        logger.info(f"apiserver connections established")
    except Exception as e:
        logger.error(f"failed to connect to apiserver: {e}")
        return None, e

    return WatcherContext(opt=opt, api_client=api_client, oss_client=oss_client), None


# uuid pattern
pattern = re.compile(r'^[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$')


def action_scan_watch_path(opt: WatcherConfig) -> List[str]:
    all_directories = os.listdir(opt.app_watchPath)

    matched_directories = list(
        map(
            lambda x: osp.join(opt.app_watchPath, x),
            filter(
                lambda x: pattern.match(x),
                filter(
                    lambda x: osp.isdir(osp.join(opt.app_watchPath, x)) and
                              osp.exists(osp.join(opt.app_watchPath, x, ".processed")),
                    all_directories
                )
            )
        )
    )

    return matched_directories


def _upload_file(client, bucket_name, local_file_path, object_name) -> Optional[Exception]:
    try:
        client.fput_object(bucket_name, object_name, local_file_path)
        logger.debug(f"file {local_file_path} uploaded as {object_name}")
        return None
    except minio.error.S3Error as exc:
        logger.error(f"error occurred while uploading {local_file_path}: {exc}")
        return Exception(f"error occurred while uploading {local_file_path}: {exc}")


def action_upload_binary(context: WatcherContext, log_path: str) -> Optional[Exception]:
    logger.debug(f"uploading binary files {log_path}")
    log_identifier = osp.basename(log_path)

    for subdir, dirs, files in os.walk(log_path):
        for file in files:
            local_file_path = os.path.join(subdir, file)
            object_name = os.path.join(log_identifier, os.path.relpath(local_file_path, log_path))
            err = _upload_file(context.oss_client, context.opt.oss_bucket, local_file_path, object_name)
            if err is not None:
                return err

    return None

def _archive_log(log_path: str, archive_path: str) -> Optional[Exception]:
    logger.debug(f"archiving {log_path}")
    try:
        shutil.move(log_path, archive_path)
        logger.debug(f"archived {log_path} to {archive_path}")
        return None
    except Exception as e:
        logger.error(f"failed to archive {log_path}: {e}")
        return e


def action_upload_log(context: WatcherContext, log_path: str) -> Optional[Exception]:
    logger.debug(f"process and uploading {log_path}")
    log_identifier = osp.basename(log_path)
    log_archive_path = osp.join(context.opt.app_watchPath, _CONFIG_ARCHIVE_PATH, log_identifier)
    err: Optional[Exception] = None

    # stage 1
    try:
        with open(osp.join(log_path, "metadata.yaml"), "r") as f:
            metadata_str = f.read()
        new_logpoint = LogPoint(
            identifier=log_identifier,
            metadata=yaml.load(io.StringIO(metadata_str), Loader=yaml.SafeLoader),
        )
    except FileNotFoundError as e:
        logger.error(f"failed to read metadata.yaml or annotation.yaml: {e}")
        err = e
    except Exception as e:
        logger.error(f"failed to parse metadata.yaml or annotation.yaml: {e}")
        err = e

    # upload binaries first
    if err is None:
        err = action_upload_binary(context, log_path)

    # upload logs to apiserver
    if err is None:
        try:
            ret = upload_logs_v1_logs_upload_post.sync_detailed(client=context.api_client, json_body=new_logpoint)
            if ret.status_code == HTTPStatus.OK:
                pass
            elif ret.status_code == HTTPStatus.BAD_REQUEST and json.loads(ret.content)["msg"] == "duplicated data point":
                pass
            else:
                logger.warning(f"failed to upload {log_path}: {ret.status_code}")
                err = Exception( f"failed to upload {log_path}: {ret.status_code}")
        except Exception as e:
            logger.error(e)
            err = e

    if err is None:
        logger.debug("successfully uploaded, archiving")
    else:
        logger.warning("failed to upload, archiving")
    _archive_log(log_path, log_archive_path)

    return err


def run(opt: WatcherConfig):
    logger.info(f"start watcher thread")

    while True:
        try:
            with DelayedKeyboardInterrupt():
                context, err = prepare_run(opt)
        except KeyboardInterrupt:
            logger.info("keyboard interrupt at staring stage, exiting")
            return

        if err is not None:
            time.sleep(10)
            continue

        _NUM_THREADS = 128
        with ThreadPoolExecutor(max_workers=_NUM_THREADS) as executor:
            try:
                while True:
                    new_logs = action_scan_watch_path(opt)
                    if len(new_logs) == 0:
                        logger.debug("no new logs, enter sleep mode")
                    else:
                        future_list = []
                        upload_start_t = time.time()
                        for idx, log_path in enumerate(new_logs):
                            future_list.append((log_path, executor.submit(action_upload_log, context, log_path)))

                        results = [(item[0], item[1].result()) for item in future_list]
                        logger.info(f"processed {len(results)} logs in {time.time() - upload_start_t}s")

                    time.sleep(30)  # 30 seconds
            except KeyboardInterrupt:
                logger.info("keyboard interrupt, exiting")
                executor.shutdown(wait=False)
                return
