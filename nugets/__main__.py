import logging
from logging import getLogger
from pathlib import Path
import sys
import wandb

from .pipeline.configs import Config
from .pipeline.pipeline import train_model
from .misc import CustomArgumentParser, create_argument_parser, configure_logging

log = getLogger(__name__)


@create_argument_parser(description="NUGETS - NeUral GEomeTry Suite")
def argument_parser(parser):
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), 
                        help="global config path")
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=None, 
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Be verbose",
        action="store_const", dest="loglevel", const=logging.INFO,
    )
    parser.add_argument(
        '--logfile',
        type=Path, help="Log file to write to (implies --verbose)",
        default=None
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    train_subparser = subparsers.add_parser("train", help="Train a model")
    train_parser(train_subparser)

    train_config_subparser = subparsers.add_parser("train_from_config", help="train a model from a config file")
    train_config_parser(train_config_subparser)

    wandb_agent_subparser = subparsers.add_parser("wandb_agent", help="run the agent for a wandb sweep")
    wandb_agent_parser(wandb_agent_subparser)

@create_argument_parser(description="train a model")
def train_parser(parser):
    from nugets.models.model import Model
    parser = Model.argument_parser(parser)
    return parser

@create_argument_parser(description="train a model from a config file")
def train_config_parser(parser):
    parser.add_argument("model_config", type=Path, help="model config path")
    return parser

@create_argument_parser(description="run the agent for a wandb sweep")
def wandb_agent_parser(parser):
    parser.add_argument("sweep_id", type=str, help="the wandb sweep id")
    parser.add_argument("--n-runs", type=int, 
                        help="number of runs to do in this sweep",
                        default=1)
    return parser

def train_from_args(args):
    from nugets.models.model import Model
    model = Model.from_args(args)
    train_model(model)

def train_from_dict(config):
    from nugets.models.model import Model
    model = Model.from_dict(config)
    train_model(model)

def train_from_config(config_file):
    from nugets.models.model import Model
    model = Model.from_config_file(config_file)
    train_model(model)

def run_from_wandb_sweep():
    from nugets.models.model import Model
    model = Model.from_dict(dict(wandb.config))
    train_model(model)
    wandb.config

def run_wandb_sweep_agent(args):
    wandb.agent(sweep_id=args.sweep_id, count=args.n_runs, function=run_from_wandb_sweep())

def main():
    parser: CustomArgumentParser = argument_parser()
    args= parser.parse_args()
    Config.load(args.config)
    config = Config.get()
    configure_logging(["nugets"], loglevel=args.loglevel if args.loglevel is not None else config.loglevel, 
                      logfile=args.logfile)
    wandb.login(key=config.wandb_key)
    # print(args)
    match args.subcommand:
        case "train":
            train_from_args(args)
        case "train_from_config":
            train_from_config(args.model_config)
        case other:
            log.error(f"Unrecognized subcommand {other}")
            sys.exit(1)   
    wandb.finish()

main()
