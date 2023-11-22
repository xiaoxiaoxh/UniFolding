# Annotation

This program presents an annotation interface to annotate experiment logs.

## Get Started

On the root directory of `unifolding`, run:

```shell
export PYTHONPATH=$PYTHONPATH:$(pwd); python ./tools/run_annotation
```

An terminal user interface will pop up. Follow the instructions.

```python
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--K', type=int, default=None, help="compare K grasp point_indices")
    parser.add_argument('--extra_filter', type=str, default=None, help="extra filter of log entries to annotate")
    parser.add_argument('--raw_log_namespace', type=str, default="experiment_real", help="namespace of original log")
    parser.add_argument('--exam_mode', action='store_true', default=False, help='launch exam mode')
    parser.add_argument('--disp_mode', action='store_true', default=False, help='launch display mode')
    parser.add_argument('--annotator', type=str, default=None)
    parser.add_argument('--api_url', type=str, default=None)
```

There are several configurable parameters, most of them a intuitive, the `api_url` should be set to the url of Service Backend Module (See ![../data_management/README.md](../data_management/README.md))
