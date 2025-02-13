import logging
from logging import getLogger
from pathlib import Path
import sys

from .misc import CustomArgumentParser, create_argument_parser, configure_logging

log = getLogger(__name__)


@create_argument_parser(description="NUGETS - NeUral GEomeTry Suite")
def argument_parser(parser):
    parser.exit_on_error = False
    parser.add_argument(
        '-d', '--debug',
        help="Print lots of debugging statements",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.WARNING,
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


def do_train(args):
    from nugets.models.model import Model
    model = Model.from_args(args)
    

def main():
    parser: CustomArgumentParser = argument_parser()
    args= parser.parse_args()
    configure_logging(["nugets"], loglevel=args.loglevel, logfile=args.logfile)
    # print(args)
    match args.subcommand:
        case "train":
            do_train(args)
        case other:
            log.error(f"Unrecognized subcommand {other}")
            sys.exit(1)   
main()
