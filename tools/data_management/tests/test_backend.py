import os
import time
import unittest

import requests
import yaml

from src.common.datamodels import LogPoint


class TestBackend(unittest.TestCase):

    def test_insert(self):
        session = requests.Session()
        dp1 = LogPoint(identifier="test.1")
        dp2 = LogPoint(identifier="test.2", annotators=['nobody'])
        dp2_1 = LogPoint(identifier="test.2", annotators=['yutong'])
        endpoint = "http://localhost:8080"
        upload_api = "/v1/logs/upload"
        print(dp1.to_json())
        print(dp2.to_json())
        print(dp2_1.to_json())
        resp = session.post(endpoint + upload_api, json=dp1.to_dict())
        print(resp.json())
        resp = session.post(endpoint + upload_api, json=dp1.to_dict())
        print(resp.json())

        log_api = "/v1/logs"
        resp = session.post(endpoint + upload_api, json=dp2.to_dict())
        print(resp.json())
        resp = session.get(endpoint + log_api, json={"identifiers": ["test.1", "test.2"]})
        print(resp.json())
        time.sleep(5)
        resp = session.post(endpoint + upload_api, json=dp2_1.to_dict())
        print(resp.json())
        resp = session.get(endpoint + log_api, json={"identifiers": ["test.1", "test.2"]})
        print(resp.json())

    def test_read(self):
        session = requests.Session()
        endpoint = "http://localhost:8080"
        log_api = "/v1/logs"
        resp = session.get(endpoint + log_api)
        print(resp.json())
        resp = session.get(endpoint + log_api, json=["test.1", "test.2"])
        print(resp.json())
        resp = session.get(endpoint + log_api, json=["test.1"])
        print(resp.json())
        resp = session.get(endpoint + log_api, json=["test.3"])
        print(resp.json())

    def test_read_all(self):
        session = requests.Session()
        endpoint = "http://localhost:8080"
        log_api = "/v1/logs"
        resp = session.get(endpoint + log_api)
        print(resp.json())

    def test_read_custom_filter(self):
        query_filter = {
            "annotators": {"$all": ["yutong"]}
        }
        session = requests.Session()
        endpoint = "http://localhost:8080"
        log_api = "/v1/logs"
        resp = session.get(endpoint + log_api, json={"extra_filter": query_filter})
        print(resp.json())
