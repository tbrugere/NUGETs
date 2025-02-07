from nugets.misc import CustomArgumentParser
import argparse


def test_grouping():
    parser = CustomArgumentParser()
    
    group0 = parser.add_argument_group(title="hello", prefix="hello", dest_group="hello")

    group1 = group0.add_argument_group(title= "world", prefix = "world", dest_group="world")
    

    # import pdb; pdb.set_trace()
    group1.add_argument("--magic", action="count")

    group2 = parser.add_argument_group(title="group2", prefix="group2", dest_group="group2")
    group2.add_argument("--magic")


    argv = ["--hello-world-magic", "--group2-magic", "one"]

    args = parser.parse_args(args=argv)
    assert args is not None
    assert args.hello.world.magic == 1
    assert args.group2.magic == "one"


def test_partial_construction():
    parser = CustomArgumentParser()

    def update_arg_1(args):
        parser.add_argument(f"--{args.arg_1}", type=int)

    parser.add_argument("--arg-1", type=str, update=update_arg_1)

    argv = ["--arg-1", "hello", "--hello", "1"]
    args = parser.parse_args(args=argv)
    assert args is not None
    assert args.arg_1 == "hello"
    assert args.hello == 1
    

def test_partial_construction_in_group():
    parser = CustomArgumentParser()


    group0 = parser.add_argument_group(title="hello", prefix="hello", dest_group="hello")

    def update_arg_1(args):
        group0.add_argument(f"--{args.arg_1}", type=int)
    group0.add_argument("--arg-1", type=str, update=update_arg_1)

    argv = ["--hello-arg-1", "hello", "--hello-hello", "1"]
    args = parser.parse_args(args=argv)
    assert args is not None
    assert args.hello.arg_1 == "hello"
    assert args.hello.hello == 1
    

