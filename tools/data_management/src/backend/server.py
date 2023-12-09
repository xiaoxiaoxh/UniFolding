from typing import Optional, Tuple
import minio
import pymongo
from loguru import logger

from src.common.config import BackendConfig
from controller import serve_forever
from src.common.datamodels import BackendContext


@logger.catch
def prepare_run(opt: BackendConfig) -> Tuple[Optional[BackendContext], Optional[Exception]]:
    try:
        oss_client = minio.Minio(endpoint=f"{opt.oss_host}:{opt.oss_port}",
                                 access_key=opt.oss_accessKey,
                                 secret_key=opt.oss_secretKey,
                                 secure=opt.oss_tls)

        buckets = oss_client.list_buckets()
        if opt.oss_bucket not in map(lambda x: x.name, buckets):
            logger.warning(f"bucket {opt.oss_bucket} does not exist, creating one")
            try:
                oss_client.make_bucket(opt.oss_bucket)
            except Exception as e:
                logger.error(f"failed to create bucket: {e}")
                return None, e
    except Exception as e:
        logger.error(f"failed to connect to oss: {e}")
        return None, e

    try:
        db_connection = pymongo.MongoClient(
            f"mongodb://{opt.db_username}:{opt.db_password}@{opt.db_host}:{opt.db_port}/")
    except Exception as e:
        logger.error(f"failed to connect to db: {e}")
        return None, e

    logger.info(f"backend connections established")
    return BackendContext(opt=opt,
                          db=db_connection,
                          oss=oss_client,
                          num_replicas=opt.app_numReplicas), None


def run(opt: BackendConfig):
    logger.info(f"start backend thread")

    context, err = prepare_run(opt)
    if err is not None:
        return err

    serve_forever(context_in=context, port=opt.api_port, host=opt.api_host)
