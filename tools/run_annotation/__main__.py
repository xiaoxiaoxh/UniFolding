import argparse
import json
import os
import sys

sys.path.insert(0, "../../")

from tools.run_annotation.io import get_io_module, AnnotatorNetworkIO

_module_root = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname((os.path.dirname(_module_root))))
_home_dir = os.path.join(_module_root, "../../")
sys.path.insert(0, _module_root)
sys.path.insert(0, _project_root)
sys.path.insert(0, _home_dir)
print("_home_dir=============================", _home_dir)

import py_cli_interaction
from loguru import logger

from common.datamodels import AnnotationConfig
from services import display, random_verify, tui_interaction


def main(args):
    opt = AnnotationConfig(
        annotator=args.annotator,
        root_dir=args.root_dir,
        K=args.K,
        raw_log_namespace=args.raw_log_namespace,
        extra_filter=args.extra_filter,
        api_url=args.api_url
    )
    return_code = 0
    # o3d.visualization.webrtc_server.enable_webrtc()
    logger.info(f"logged in as {args.annotator}")
    io_module = get_io_module(opt)

    while True:
        unprocessed_log_entries, processed_log_entries, corrupted_log_entries, err = io_module.scan_log_dir()
        unprocessed_log_entries.sort()
        processed_log_entries.sort()

        if err is not None:
            logger.error(f'error: {err}')
            return_code = 1
            break

        try:
            if args.exam_mode:
                err = random_verify(opt, processed_log_entries)
            elif args.disp_mode:
                err = display(opt, unprocessed_log_entries + processed_log_entries)
            else:
                err = tui_interaction(opt, unprocessed_log_entries)
            if err is not None:
                logger.error(f'error: {err}')
                return_code = 1
                break

        except KeyboardInterrupt:
            logger.info('user interrupt')
            break

        except Exception as e:
            logger.error(e)

        finally:
            proceed = py_cli_interaction.must_parse_cli_bool('proceed?')
            if proceed:
                continue
            else:
                break

    return return_code


def as_complete(args):
    err = None
    if args.root_dir is None:
        args.root_dir, err = AnnotatorNetworkIO.select_log_dir()
    assert (args.root_dir is not None) and (err is None), ValueError('root_dir is not specified')

    if args.K is None:
        args.K = py_cli_interaction.must_parse_cli_int("compare K grasp point_indices", default_value=16)

    if args.annotator is None:
        args.annotator = os.environ.get('UNIFOLDING_ANNOTATOR', 'nobody')

    if args.api_url is None:
        args.api_url = py_cli_interaction.must_parse_cli_string("input api url", default_value="http://127.0.0.1:8080")

    if args.extra_filter is not None:
        try:
            args.extra_filter = json.loads(args.extra_filter)
        except Exception:
            raise ValueError('extra_filter is not a valid json string')
    return args


def entrypoint(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--K', type=int, default=None, help="compare K grasp point_indices")
    parser.add_argument('--extra_filter', type=str, default=None, help="extra filter of log entries to annotate")
    parser.add_argument('--raw_log_namespace', type=str, default="experiment_real", help="namespace of original log")
    parser.add_argument('--exam_mode', action='store_true', default=False, help='launch exam mode')
    parser.add_argument('--disp_mode', action='store_true', default=False, help='launch display mode')
    parser.add_argument('--annotator', type=str, default=None)
    parser.add_argument('--api_url', type=str, default=None)

    args = parser.parse_args(argv)

    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    return main(as_complete(args))


if __name__ == '__main__':
    import sys

    exit(entrypoint(sys.argv[1:]))
