
from .misc import CustomArgumentParser, create_argument_parser


@create_argument_parser(description="NUGETS - NeUral GEomeTry Suite")
def argument_parser(parser):

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    train_subparser = subparsers.add_parser("train", help="Train a model")
    train_parser(train_subparser)


@create_argument_parser(description="train a model")
def train_parser(parser):
    from nugets.models.model import Model
    parser = Model.argument_parser(parser)
    return parser


def do_train(args):
    pass
    

def main():
    parser: CustomArgumentParser = argument_parser()
    args= parser.parse_args()

    

main()
