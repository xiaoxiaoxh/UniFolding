import click
from src.common.config import (
    watcher_get_cli_parser,
    watcher_get_default_config,
    watcher_save_config,
    watcher_load_config,
    WatcherConfig
)
from server import run
from loguru import logger


@click.group()
@click.pass_context
def cli(ctx):
    pass


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def init(ctx):
    import sys
    argv = sys.argv[2:]
    logger.info("init with args: {}", argv)
    args = watcher_get_cli_parser().parse_args(argv)
    v = watcher_get_default_config()
    if args.config is not None:
        v.set_config_file(args.config)
        try:
            v.read_in_config()
        except Exception as e:
            logger.debug(e)
    v.bind_args(vars(args))

    err = watcher_save_config(v, path=args.config)
    if err is not None:
        logger.error(err)


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def serve(ctx):
    v, err = watcher_load_config(argv=[])
    opt = WatcherConfig().from_vyper(v)
    run(opt)


if __name__ == '__main__':
    cli()
