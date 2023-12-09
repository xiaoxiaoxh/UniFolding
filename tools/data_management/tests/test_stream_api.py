import json

import requests
from loguru import logger
from tools.data_management.src.client import Client
from tools.data_management.src.client.api.default import get_log_stream_v1_log_stream_get
from tools.data_management.src.client.models import LogRequest

BASE_URL = "http://100.99.96.202:8080"
api_url = BASE_URL + "/v1/log_stream"
session = requests.Session()
resp = session.get(api_url)
res = [json.loads(jline) for jline in resp.content.splitlines()]
logger.debug(f"number of archives: {len(res)}")

c = Client(base_url=BASE_URL, verify_ssl=False)
resp = get_log_stream_v1_log_stream_get.sync_detailed(client=c, json_body=LogRequest())
res = [json.loads(jline) for jline in resp.content.splitlines()]

