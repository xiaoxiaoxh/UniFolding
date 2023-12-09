import datetime
import json
import os
import threading
import time
from typing import Optional

import pymongo
import minio
import uvicorn
from bson import json_util
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from src.common.datamodels import BackendContext, LogPoint, LogRequest, AnnotationPoint

controller = FastAPI()
context: Optional[BackendContext] = None


def make_response(status_code, **kwargs):
    data = {'code': status_code, 'timestamp': time.time()}
    data.update(**kwargs)
    json_compatible_data = jsonable_encoder(data)
    resp = JSONResponse(content=json_compatible_data, status_code=status_code)
    return resp


@controller.get("/")
def root():
    return RedirectResponse(url='/docs')


# >>>>>>>>> commented out to disable this api <<<<<<<<<
# @controller.get("/v1/num_replicas")
# def get_num_replicas():
#     global context
#     return make_response(200, num_replicas=context.num_replicas)
# >>>>>>>>> commented out to disable this api <<<<<<<<<

# >>>>>>>>> commented out to disable this api <<<<<<<<<
# @controller.post("/v1/num_replicas")
# def set_num_replicas(num_replicas: int):
#     global context
#     context.num_replicas = num_replicas
# >>>>>>>>> commented out to disable this api <<<<<<<<<


# >>>>>>>>> commented out to disable this api <<<<<<<<<
# @controller.post("/v1/users")
# def create_user(user_id: str):
#     return make_response(500, msg="NotImplementedError")  # TODO implement
# >>>>>>>>> commented out to disable this api <<<<<<<<<
@controller.get("/v1/proxy/{identifier}/")
def proxy_oss_get_object(identifier: str, rel_path: str):
    global context
    if context.oss is None:
        return make_response(500, msg="OSS not configured")

    # return the object file as a response
    try:
        # Get the requested file as a stream from MinIO
        response = context.oss.get_object(context.opt.oss_bucket, f"{identifier}/{rel_path}")
        return StreamingResponse(response.stream(), media_type="application/octet-stream")
    except minio.error.S3Error as e:
        return make_response(500, msg=str(e))


@controller.post("/v1/logs/upload")
def upload_logs(log_point: LogPoint):
    global context
    col = context.db[context.opt.db_database][context.opt.const_db_collection]
    res = col.find_one({"identifier": log_point.identifier})
    if res is not None:
        col.update_one({"identifier": log_point.identifier}, {"$set": log_point.to_dict()})
        return make_response(200, msg="success (duplicated data point)")
    try:
        res = col.insert_one(log_point.to_dict())
        return make_response(200, msg="success", id=str(res.inserted_id))
    except Exception as e:
        return make_response(500, msg=str(e))


# @controller.post("/v1/annotations/{identifier}")
# def upload_annotation(identifier: str, anno_point: AnnotationPoint):
#     global context
#     col = context.db[context.opt.db_database][context.opt.const_db_collection]
#     res = col.find_one({"identifier": identifier})
#     if res is None:
#         return make_response(404, msg="identifier not found")
#
#     idx = int(time.time() * 1000)
#     res['annotations'][f"{anno_point.annotator}.{idx}"] = anno_point.annotation
#     res['annotators'].append(anno_point.annotator) if anno_point.annotator not in res['annotators'] else None
#     col.update_one(
#         {"identifier": identifier}, {"$set": {"annotations": res['annotations'], "annotators": res['annotators']}}
#     )


@controller.post("/v1/locks/{identifier}")
def lock_log(identifier: str):
    global context
    col = context.db[context.opt.db_database][context.opt.const_db_collection]
    res = col.find_one({"identifier": identifier})
    if res is None:
        return make_response(404, msg="identifier not found")
    locked = res.get('_locked', None)
    if locked is not None and (datetime.datetime.now().timestamp() - float(locked)) < 600:
        return make_response(403, msg="locked")
    else:
        try:
            _ = col.update_one({"identifier": identifier}, {"$set": {"_locked": datetime.datetime.now().timestamp()}})
            return make_response(200, msg="success")
        except Exception as e:
            return make_response(500, msg=str(e))


@controller.delete("/v1/locks/{identifier}")
def unlock_log(identifier: str):
    global context
    col = context.db[context.opt.db_database][context.opt.const_db_collection]
    res = col.find_one({"identifier": identifier})
    if res is None:
        return make_response(404, msg="identifier not found")
    try:
        _ = col.update_one({"identifier": identifier}, {"$unset": {"_locked": ""}})
        return make_response(200, msg="success")
    except Exception as e:
        return make_response(500, msg=str(e))


@controller.put("/v1/processed/{identifier}")
def set_processed_flag(identifier: str, value: int):
    global context
    col = context.db[context.opt.db_database][context.opt.const_db_collection]
    res = col.find_one({"identifier": identifier})
    if res is None:
        return make_response(404, msg="identifier not found")

    try:
        _ = col.update_one({"identifier": identifier}, {"$set": {"_processed": value}})
        return make_response(200, msg="success")
    except Exception as e:
        return make_response(500, msg=str(e))


@controller.get("/v1/processed/{identifier}")
def get_processed_flag(identifier: str):
    global context
    col = context.db[context.opt.db_database][context.opt.const_db_collection]
    res = col.find_one({"identifier": identifier})
    if res is None:
        return make_response(404, msg="identifier not found")

    return make_response(200, value=res.get('_processed', 0))


@controller.post("/v1/logs")
def get_logs(req: LogRequest):
    global context
    col = context.db[context.opt.db_database][context.opt.const_db_collection]

    if req.identifiers is not None:
        if req.extra_filter is None:
            req.extra_filter = {}
        req.extra_filter["identifier"] = {"$in": req.identifiers}

    try:
        res = list(col.find(req.extra_filter))
        logs = json_util._json_convert(res)
        return make_response(200, msg="success", logs=logs)
    except Exception as e:
        return make_response(500, msg=str(e))


@controller.get("/v1/logs")
def get_logs_legal(req: LogRequest):
    return get_logs(req)


def logs_stream(cursor: Optional[pymongo.cursor.Cursor]):
    global context
    if cursor is None:
        return ""
    for log in cursor:
        yield json.dumps(json_util._json_convert(log)) + "\n"


@controller.post("/v1/log_stream")
def get_log_stream(req: LogRequest) -> StreamingResponse:
    global context
    col = context.db[context.opt.db_database][context.opt.const_db_collection]

    if req.identifiers is not None:
        if req.extra_filter is None:
            req.extra_filter = {}
        req.extra_filter["identifier"] = {"$in": req.identifiers}

    try:
        res = col.find(req.extra_filter)
        return StreamingResponse(logs_stream(res), media_type="text/jsonl")
    except Exception as e:
        return StreamingResponse(logs_stream(None), media_type="text/jsonl")


@controller.get("/v1/log_stream")
def get_log_stream_legal(req: LogRequest) -> StreamingResponse:
    return get_log_stream(req)


def serve_forever(context_in: BackendContext, port: int, host: str = '0.0.0.0'):
    global context
    context = context_in
    try:
        thread = threading.Thread(target=uvicorn.run, kwargs={'app': controller, 'port': port, 'host': host})
        thread.start()
        while True:
            time.sleep(86400)
        # uvicorn.run(app=controller, port=cfg.api_port, host=cfg.api_interface)
    except KeyboardInterrupt:
        print(f"got KeyboardInterrupt")
        os._exit(1)
