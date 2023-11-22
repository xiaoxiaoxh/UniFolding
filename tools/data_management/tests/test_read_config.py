import os
import unittest
from src.common.config import watcher_load_config, watcher_get_default_config, watcher_save_config, \
    watcher_get_cli_parser


class TestWatcherConfig(unittest.TestCase):
    def test_load(self):
        v, err = watcher_load_config(
            argv=["--config", "manifests/watcher.yaml", "--oss.host", "oss-cn-shanghai.aliyuncs.com"])
        print(v.get("api.url"))
        print(v.get("oss.host"))
        print(v.get("oss.port"))
        print(v.get("oss.accessKey"))
        print(v.get("oss.secretKey"))
        print(v.get("oss.bucket"))
        self.assertIsNone(err)

        os.environ["UNIFOLDING_OSS_PORT"] = "19000"  # Usually set externally
        v, err = watcher_load_config(argv=[])
        print(v.get("api.url"))
        print(v.get("oss.host"))
        print(v.get("oss.port"))
        print(v.get("oss.accessKey"))
        print(v.get("oss.secretKey"))
        print(v.get("oss.bucket"))
        self.assertIsNone(err)

        pass

    def test_save_default(self):
        parser = watcher_get_cli_parser()
        args = parser.parse_args(["--oss.host", "100.99.96.202", "--oss.accessKey", "unifolding"])
        v = watcher_get_default_config()
        if args.config is not None:
            v.set_config_file(args.config)
            try:
                v.read_in_config()
            except Exception as e:
                print(e)
        v.bind_args(vars(args))

        err = watcher_save_config(v, path=args.config)
        self.assertIsNone(err)
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
