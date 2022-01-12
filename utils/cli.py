import click
from typing import Callable, Optional, Tuple

from .log import ExperimentLog
from ..agents import Agent
from ..config import Config
from ..run import eval_agent, train_agent, random_agent, SAVE_FILE_DEFAULT


@click.group()
@click.option('--gpu', required=False, type=int)
@click.option('--seed', type=int, default=None)
@click.pass_context
def rainy_cli(ctx: dict, gpu: Tuple[int], seed: Optional[int]) -> None:
    ctx.obj['gpu'] = gpu
    ctx.obj['config'].seed = seed


@rainy_cli.command()
@click.pass_context
@click.option('--comment', type=str, default=None)
@click.option('--prefix', type=str, default='')
def train(ctx: dict, comment: Optional[str], prefix: str) -> None:
    c = ctx.obj['config']
    scr = ctx.obj['script_path']
    if scr:
        c.logger.set_dir_from_script_path(scr, comment=comment, prefix=prefix)
    c.logger.set_stderr()
    ag = ctx.obj['make_agent'](c)
    train_agent(ag)
    print("random play: {}, trained: {}".format(ag.random_episode(), ag.eval_episode()))


@rainy_cli.command()
@click.option('--save', is_flag=True)
@click.option('--render', is_flag=True)
@click.option('--replay', is_flag=True)
@click.option('--action-file', type=str, default='random-actions.json')
@click.pass_context
def random(ctx: dict, save: bool, render: bool, replay: bool, action_file: str) -> None:
    c = ctx.obj['config']
    ag = ctx.obj['make_agent'](c)
    action_file = fname if save else None
    random_agent(ag, render=render, replay=replay, action_file=action_file)


@rainy_cli.command()
@click.argument('logdir', required=True, type=str)
@click.option('--model', type=str, default=SAVE_FILE_DEFAULT)
@click.option('--render', is_flag=True)
@click.option('--replay', is_flag=True)
@click.option('--action-file', type=str, default='best-actions.json')
@click.pass_context
def eval(ctx: dict, logdir: str, model: str, render: bool, replay: bool, action_file: str) -> None:
    c = ctx.obj['config']
    ag = ctx.obj['make_agent'](c)
    eval_agent(
        ag,
        logdir,
        load_file_name=model,
        render=render,
        replay=replay,
        action_file=action_file
    )


@rainy_cli.command()
@click.option('--log-dir', type=str)
@click.option('--vi-mode', is_flag=True)
@click.pass_context
def ipython(ctx: dict, log_dir: Optional[str], vi_mode: bool) -> None:
    config, make_agent = ctx.obj['config'], ctx.obj['make_agent']  # noqa
    if log_dir is not None:
        log = ExperimentLog(log_dir)  # noqa
    else:
        open_log = ExperimentLog  # noqa
    try:
        from ptpython.ipython import embed
        del ctx, log_dir
        import rainy  # noqa
        embed(vi_mode=vi_mode)
    except ImportError:
        print("To use ipython mode, install ipython and ptpython first.")


def run_cli(
        config: Config,
        agent_gen: Callable[[Config], Agent],
        script_path: Optional[str] = None
) -> rainy_cli:
    obj = {
        'config': config,
        'make_agent': agent_gen,
        'script_path': script_path
    }
    rainy_cli(obj=obj)

