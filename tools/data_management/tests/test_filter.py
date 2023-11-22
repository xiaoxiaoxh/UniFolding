import json

__filter_str__ = """
{"$and": [{"metadata.experiment_real.tag": {"$exists": "true", "$eq": "tshirt_short_action14_real"}}, {"metadata.experiment_real.episode_idx": {"$exists": "true", "$gte": 0}}, {"metadata.experiment_real.episode_idx": {"$exists": "true", "$lt": 1}}]}
"""

__filter_json__ = {
    "extra_filter": json.loads(__filter_str__)
}

import requests
import tqdm
from loguru import logger

api_url = "http://127.0.0.1:8080/v1/logs"
session = requests.Session()
resp = session.get(api_url, json=__filter_json__)
res = json.loads(resp.json()["logs_json"])
logger.debug(f"number of archives: {len(res)}")

