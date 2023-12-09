import argparse
import dataclasses
from typing import Optional, Dict, Any, Tuple, List

import yaml
from vyper import Vyper
import os
import os.path as osp
from loguru import logger

_HOME = os.path.expanduser('~')
_CONFIG_PROJECT_NAME = "unifolding"

_CONFIG_WATCHER_CONFIG_NAME = "watcher"
_CONFIG_BACKEND_CONFIG_NAME = "backend"

_CONFIG_WATCHER_WATCH_PATH = "/data"


@dataclasses.dataclass
class WatcherConfig:
    api_url: str = dataclasses.field(default="http://api.example.com:8080")
    oss_host: str = dataclasses.field(default="oss.example.com")
    oss_port: int = dataclasses.field(default=9000)
    oss_tls: bool = dataclasses.field(default=False)
    oss_accessKey: str = dataclasses.field(default=_CONFIG_PROJECT_NAME)
    oss_secretKey: str = dataclasses.field(default=_CONFIG_PROJECT_NAME)
    oss_bucket: str = dataclasses.field(default=_CONFIG_PROJECT_NAME)
    app_watchPath: str = dataclasses.field(default=_CONFIG_WATCHER_WATCH_PATH)

    @logger.catch
    def from_dict(self, d: Dict[str, Any]):
        self.api_url = d["api"]["url"]
        self.oss_host = d["oss"]["host"]
        self.oss_port = d["oss"]["port"]
        self.oss_tls = d["oss"]["tls"]
        self.oss_accessKey = d["oss"]["accessKey"]
        self.oss_secretKey = d["oss"]["secretKey"]
        self.oss_bucket = d["oss"]["bucket"]
        self.app_watchPath = d["app"]["watchPath"]
        return self

    @logger.catch
    def from_vyper(self, v: Vyper):
        self.api_url = v.get("api.url")
        self.oss_host = v.get("oss.host")
        self.oss_port = v.get("oss.port")
        self.oss_tls = v.get("oss.tls")
        self.oss_accessKey = v.get("oss.accessKey")
        self.oss_secretKey = v.get("oss.secretKey")
        self.oss_bucket = v.get("oss.bucket")
        self.app_watchPath = v.get("app.watchPath")
        return self

    def to_dict(self):
        return {
            "api": {
                "url": self.api_url
            },
            "oss": {
                "host": self.oss_host,
                "port": self.oss_port,
                "tls": self.oss_tls,
                "accessKey": self.oss_accessKey,
                "secretKey": self.oss_secretKey,
                "bucket": self.oss_bucket
            },
            "app": {
                "watchPath": self.app_watchPath,
            },
        }


@dataclasses.dataclass
class BackendConfig:
    app_numReplicas: int = dataclasses.field(default=1)
    api_host: str = dataclasses.field(default="0.0.0.0")
    api_port: int = dataclasses.field(default=8080)
    db_host: str = dataclasses.field(default="mongodb.example.com")
    db_port: int = dataclasses.field(default=27017)
    db_database: str = dataclasses.field(default=_CONFIG_PROJECT_NAME)
    db_username: str = dataclasses.field(default=_CONFIG_PROJECT_NAME)
    db_password: str = dataclasses.field(default=_CONFIG_PROJECT_NAME)
    oss_host: str = dataclasses.field(default="oss.example.com")
    oss_port: int = dataclasses.field(default=9000)
    oss_tls: bool = dataclasses.field(default=False)
    oss_accessKey: str = dataclasses.field(default=_CONFIG_PROJECT_NAME)
    oss_secretKey: str = dataclasses.field(default=_CONFIG_PROJECT_NAME)
    oss_bucket: str = dataclasses.field(default=_CONFIG_PROJECT_NAME)

    const_db_collection: str = dataclasses.field(default="logs")

    @logger.catch
    def from_dict(self, d: Dict[str, Any]):
        self.app_numReplicas = d["app"]["numReplicas"]
        self.api_host = d["api"]["host"]
        self.api_port = d["api"]["port"]
        self.db_host = d["db"]["host"]
        self.db_port = d["db"]["port"]
        self.db_database = d["db"]["database"]
        self.db_username = d["db"]["username"]
        self.db_password = d["db"]["password"]
        self.oss_host = d["oss"]["host"]
        self.oss_port = d["oss"]["port"]
        self.oss_tls = d["oss"]["tls"]
        self.oss_accessKey = d["oss"]["accessKey"]
        self.oss_secretKey = d["oss"]["secretKey"]
        self.oss_bucket = d["oss"]["bucket"]
        return self

    @logger.catch
    def from_vyper(self, v: Vyper):
        self.app_numReplicas = v.get("app.numReplicas")
        self.api_host = v.get("api.host")
        self.api_port = v.get_int("api.port")
        self.db_host = v.get("db.host")
        self.db_port = v.get_int("db.port")
        self.db_database = v.get("db.database")
        self.db_username = v.get("db.username")
        self.db_password = v.get("db.password")
        self.oss_host = v.get("oss.host")
        self.oss_port = v.get_int("oss.port")
        self.oss_tls = v.get_bool("oss.tls")
        self.oss_accessKey = v.get("oss.accessKey")
        self.oss_secretKey = v.get("oss.secretKey")
        self.oss_bucket = v.get("oss.bucket")
        return self

    def to_dict(self):
        return {
            "app": {
                "numReplicas": self.app_numReplicas,
            },
            "api": {
                "host": self.api_host,
                "port": self.api_port,
            },
            "db": {
                "host": self.db_host,
                "port": self.db_port,
                "database": self.db_database,
                "username": self.db_username,
                "password": self.db_password,
            },
            "oss": {
                "host": self.oss_host,
                "port": self.oss_port,
                "tls": self.oss_tls,
                "accessKey": self.oss_accessKey,
                "secretKey": self.oss_secretKey,
                "bucket": self.oss_bucket,
            },
        }


def watcher_get_default_config() -> Vyper:
    """
    watcher
      api:
        url: http://api.example.com:8080
      oss:
        host: oss.example.com
        port: 9000
        tls: false
        access_key: unifolding
        secret_key: unifolding
        bucket: unifolding
    """
    v = Vyper()
    _DEFAULT = WatcherConfig()
    v.set_default("api.url", _DEFAULT.api_url)
    v.set_default("oss.host", _DEFAULT.oss_host)
    v.set_default("oss.port", _DEFAULT.oss_port)
    v.set_default("oss.tls", _DEFAULT.oss_tls)
    v.set_default("oss.accessKey", _DEFAULT.oss_accessKey)
    v.set_default("oss.secretKey", _DEFAULT.oss_secretKey)
    v.set_default("oss.bucket", _DEFAULT.oss_bucket)
    v.set_default("app.watchPath", _DEFAULT.app_watchPath)
    return v


def backend_get_default_config() -> Vyper:
    """
    backend:
      app:
        numReplicas: 1
      api:
        port: 8080
        host: '0.0.0.0'
      db:
        host: mongodb.example.com
        port: 27017
        username: unifolding
        password: unifolding
        database: unifolding
      oss:
        host: oss.example.com
        port: 9000
        tls: false
        accessKey: unifolding
        secretKey: unifolding
        bucket: unifolding
    """
    v = Vyper()
    _DEFAULT = BackendConfig()
    v.set_default("app.numReplicas", _DEFAULT.app_numReplicas)
    v.set_default("api.host", _DEFAULT.api_host)
    v.set_default("api.port", _DEFAULT.api_port)
    v.set_default("db.host", _DEFAULT.db_host)
    v.set_default("db.port", _DEFAULT.db_port)
    v.set_default("db.database", _DEFAULT.db_database)
    v.set_default("db.username", _DEFAULT.db_username)
    v.set_default("db.password", _DEFAULT.db_password)
    v.set_default("oss.host", _DEFAULT.oss_host)
    v.set_default("oss.port", _DEFAULT.oss_port)
    v.set_default("oss.tls", _DEFAULT.oss_tls)
    v.set_default("oss.accessKey", _DEFAULT.oss_accessKey)
    v.set_default("oss.secretKey", _DEFAULT.oss_secretKey)
    v.set_default("oss.bucket", _DEFAULT.oss_bucket)

    return v


def watcher_get_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config file path", default=None)
    parser.add_argument("--api.url", type=str, help="api url")
    parser.add_argument("--oss.host", type=str, help="oss host")
    parser.add_argument("--oss.port", type=int, help="oss port")
    parser.add_argument("--oss.tls", type=bool, help="oss tls")
    parser.add_argument("--oss.accessKey", type=str, help="oss access key")
    parser.add_argument("--oss.secretKey", type=str, help="oss secret key")
    parser.add_argument("--oss.bucket", type=str, help="oss bucket")
    parser.add_argument("--app.watchPath", type=str, help="watch path")
    return parser


def backend_get_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config file path", default=None)
    parser.add_argument("--app.numReplicas", type=int, help="app num replicas")
    parser.add_argument("--api.host", type=str, help="api host")
    parser.add_argument("--api.port", type=int, help="api port")
    parser.add_argument("--db.host", type=str, help="db host")
    parser.add_argument("--db.port", type=int, help="db port")
    parser.add_argument("--db.database", type=str, help="db database")
    parser.add_argument("--db.username", type=str, help="db username")
    parser.add_argument("--db.password", type=str, help="db password")
    parser.add_argument("--oss.host", type=str, help="oss host")
    parser.add_argument("--oss.port", type=int, help="oss port")
    parser.add_argument("--oss.tls", type=bool, help="oss tls")
    parser.add_argument("--oss.accessKey", type=str, help="oss access key")
    parser.add_argument("--oss.secretKey", type=str, help="oss secret key")
    parser.add_argument("--oss.bucket", type=str, help="oss bucket")

    return parser


def watcher_load_config(argv: List[str]) -> Tuple[Vyper, Optional[Exception]]:
    parser = watcher_get_cli_parser()
    args = parser.parse_args(argv)

    v = watcher_get_default_config()
    v.set_config_name(_CONFIG_WATCHER_CONFIG_NAME)
    v.set_config_type("yaml")
    v.add_config_path(f"/etc/{_CONFIG_PROJECT_NAME}")
    v.add_config_path(osp.join(_HOME, f".{_CONFIG_PROJECT_NAME}"))
    v.add_config_path(".")
    if args.config is not None:
        v.set_config_file(args.config)
    try:
        v.merge_in_config()
        logger.debug(f"load config form : {v._config_file}")
    except FileNotFoundError:
        v = watcher_get_default_config()
        logger.warning(f"config file not found")

    v.set_env_prefix(_CONFIG_PROJECT_NAME.upper())
    v.set_env_key_replacer(".", "_")

    v.bind_args(vars(args))
    v.bind_env("api.url")
    v.bind_env("oss.host")
    v.bind_env("oss.port")
    v.bind_env("oss.tls")
    v.bind_env("oss.accessKey")
    v.bind_env("oss.secretKey")
    v.bind_env("oss.bucket")
    v.bind_env("app.watchPath")

    logger.debug(f"watcher config: {WatcherConfig().from_vyper(v).to_dict()}")

    return v, None


def backend_load_config(argv: List[str]) -> Tuple[Vyper, Optional[Exception]]:
    parser = backend_get_cli_parser()
    args = parser.parse_args(argv)

    v = backend_get_default_config()
    v.set_config_name(_CONFIG_BACKEND_CONFIG_NAME)
    v.set_config_type("yaml")
    v.add_config_path(f"/etc/{_CONFIG_PROJECT_NAME}")
    v.add_config_path(osp.join(_HOME, f".{_CONFIG_PROJECT_NAME}"))
    v.add_config_path(".")
    if args.config is not None:
        v.set_config_file(args.config)
    try:
        v.merge_in_config()
        logger.debug(f"load config form : {v._config_file}")
    except FileNotFoundError:
        v = backend_get_default_config()
        logger.warning(f"config file not found")

    v.set_env_prefix(_CONFIG_PROJECT_NAME.upper())
    v.set_env_key_replacer(".", "_")

    v.bind_args(vars(args))
    v.bind_env("app.numReplicas")
    v.bind_env("api.host")
    v.bind_env("api.port")
    v.bind_env("db.host")
    v.bind_env("db.port")
    v.bind_env("db.database")
    v.bind_env("db.username")
    v.bind_env("db.password")
    v.bind_env("oss.host")
    v.bind_env("oss.port")
    v.bind_env("oss.tls")
    v.bind_env("oss.accessKey")
    v.bind_env("oss.secretKey")
    v.bind_env("oss.bucket")

    logger.debug(f"backend config: {BackendConfig().from_vyper(v).to_dict()}")

    return v, None


def watcher_save_config(v: Vyper, path: str = None) -> Optional[Exception]:
    if path is None:
        path = osp.join(_HOME, f".{_CONFIG_PROJECT_NAME}", f"{_CONFIG_WATCHER_CONFIG_NAME}.yaml")

    _DIR = osp.dirname(path)
    if not osp.exists(_DIR):
        os.makedirs(_DIR, exist_ok=True)

    _VALUE = WatcherConfig().from_vyper(v).to_dict()

    logger.debug(f"save path: {path}")
    logger.debug(f"save config: {_VALUE}")

    with open(path, "w") as f:
        yaml.dump(_VALUE, f, default_flow_style=False)

    return None


def backend_save_config(v: Vyper, path: str = None) -> Optional[Exception]:
    if path is None:
        path = osp.join(_HOME, f".{_CONFIG_PROJECT_NAME}", f"{_CONFIG_BACKEND_CONFIG_NAME}.yaml")

    _DIR = osp.dirname(path)
    if not osp.exists(_DIR):
        os.makedirs(_DIR, exist_ok=True)

    _VALUE = BackendConfig().from_vyper(v).to_dict()

    logger.debug(f"save path: {path}")
    logger.debug(f"save config: {_VALUE}")

    with open(path, "w") as f:
        yaml.dump(_VALUE, f, default_flow_style=False)

    return None
