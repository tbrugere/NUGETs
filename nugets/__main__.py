import logging
from logging import getLogger
from pathlib import Path
import sys

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


@create_argument_parser(description="train a model")
def train_parser(parser):
    from nugets.models.model import Model
    parser = Model.argument_parser(parser)
    return parser

@create_argument_parser(description="train a model from a config file")
def train_config_parser(parser):
    from nugets.models.model import Model
    parser.add_argument("model_config", type=Path, help="model config path")
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

def main():
    parser: CustomArgumentParser = argument_parser()
    args= parser.parse_args()
    Config.load(args.config)
    config = Config.get()
    configure_logging(["nugets"], loglevel=args.loglevel if args.loglevel is not None else config.loglevel, 
                      logfile=args.logfile)
    # print(args)
    match args.subcommand:
        case "train":
            train_from_args(args)
        case "train_from_config":
            train_from_config(args.model_config)
        case other:
            log.error(f"Unrecognized subcommand {other}")
            sys.exit(1)   
main()
