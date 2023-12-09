import os
import unittest

import pymongo
import yaml

from src.common.datamodels import LogPoint

cfg = yaml.load(open("manifests/backend.yaml", "r"), Loader=yaml.FullLoader)
cfg['db']['host'] = os.environ['UNIFOLDING_DB_HOST']


class TestDocumentConfig(unittest.TestCase):
    def test_init_db(self):
        client = pymongo.MongoClient(
            f"mongodb://{cfg['db']['username']}:{cfg['db']['password']}@{cfg['db']['host']}:{cfg['db']['port']}/"
        )
        print(client.list_database_names())
        db = client[cfg['db']['database']]
        col = db.list_collection_names()
        print(col)

    def test_insert(self):
        client = pymongo.MongoClient(
            f"mongodb://{cfg['db']['username']}:{cfg['db']['password']}@{cfg['db']['host']}:{cfg['db']['port']}/"
        )
        print(client.list_database_names())
        db = client[cfg['db']['database']]
        col = db['test']
        dp = LogPoint(identifier="test")
        col.insert_one(dp.to_dict())
        print(col.find_one({"identifier": "test"}))
