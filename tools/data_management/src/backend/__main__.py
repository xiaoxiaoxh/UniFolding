import click
from src.common.config import (
    backend_get_cli_parser,
    backend_get_default_config,
    backend_save_config,
    backend_load_config,
    BackendConfig
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
    args = backend_get_cli_parser().parse_args(argv)
    v = backend_get_default_config()
    if args.config is not None:
        v.set_config_file(args.config)
        try:
            v.read_in_config()
        except Exception as e:
            logger.debug(e)
    v.bind_args(vars(args))

    err = backend_save_config(v, path=args.config)
    if err is not None:
        logger.error(err)


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def serve(ctx):
    v, err = backend_load_config(argv=[])
    opt = BackendConfig().from_vyper(v)
    run(opt)


if __name__ == '__main__':
    cli()
