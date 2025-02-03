
from .misc import create_argument_parser


@create_argument_parser(description="NUGETS - NeUral GEomeTry Suite")
def argument_parser(parser):

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")


@create_argument_parser(description="train a model")
def train_parser(parser):
    parser.add_argument("--task", type=str, required=True, help="The task to train on")
    parser.add_argument("--backbone", type=str, required=True, help="The backbone to use")
    parser.add_argument("--batch-size", type=int, required=True, help="The batch size")
    parser.add_argument("--learning-rate", type=float, required=True, help="The learning rate")


def main():
    

if 
