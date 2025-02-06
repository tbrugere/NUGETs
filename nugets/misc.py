"""Miscellaneous utilities"""

from typing import Any, Callable, TypeVar, Protocol, ParamSpec
from argparse import ArgumentParser
import functools as ft
from io import BytesIO
from json import dumps as json_dumps

P = ParamSpec("P")
T = TypeVar("T")

def take_argument_annotation_from(this: Callable[P, Any]) \
        -> Callable[[Callable[..., T]], Callable[P, T]]:
    """Take the argument annotations from another function

    Decorator stating that the function it decorates 
    should have the same annotations as the function passed as argument.

    Inspired from https://stackoverflow.com/a/71262408/4948719

    """
    def decorator(real_function: Callable) -> Callable[P, T]:
        return_type ={"return": real_function.__annotations__["return"]} if "return" in real_function.__annotations__ else {}
        real_function.__annotations__ = {**this.__annotations__, **return_type}
        return real_function #type: ignore 
    return decorator #type: ignore

class CreateArgumentParserReturnSignature(Protocol): #noqa: D101
    def __call__(self, parser: ArgumentParser|None = None) -> ArgumentParser: ... #noqa: D102

@take_argument_annotation_from(ArgumentParser)
def create_argument_parser(**argparse_kwargs) \
        -> Callable[[Callable[[ArgumentParser], None]], 
                    CreateArgumentParserReturnSignature]:
    """Decorator that potentially creates an argument parser and passes it down

    This is a helper function that allows to have a function that either

    - populates an already existing parser 
        (if passed as argument, generally for use with subparsers)
    - creates an all new parser

    It decorates a function that takes a single argument, an ArgumentParser, 
    and populates it.

    usage:

    .. code-block:: python

        # @create_argument_parser
        # accepts all the arguments of ArgumentParser. 
        # They will be passed to the ArgumentParser constructor 
        # if a new parser is created
        @create_argument_parser(description="This is the description") 
        def my_argument_parser(parser: ArgumentParser):
            parser.add_argument("--my-arg", help="This is the help")

        specific_parser = my_argument_parser()
        # or
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()
        my_argument_parser(subparsers.add_parser("my_subcommand"))

    .. note:: 

        The input function should only take a single argument parser, 
        and does not need to return anything. 
        The output function will return the parser.

    """

    def decorator(func: Callable[[ArgumentParser], None]) \
            -> CreateArgumentParserReturnSignature:
        @ft.wraps(func)
        def wrapper(parser: ArgumentParser|None = None) -> ArgumentParser:
            if parser is None:
                parser = ArgumentParser(**argparse_kwargs)
            func(parser)
            return parser
        return wrapper
    return decorator

def dict_to_bytes(d: dict[str, Any]):
    """

    Takes a dict where keys are strings, and values are jsonable, 
    and outputs an order-independent set of bytes for hashing purposes

    """

    res = BytesIO()

    tuples = [(k, v) for k, v in d.items()]
    tuples.sort()

    dumps = ft.partial(json_dumps, separators=(',', ':'), sort_keys=True)

    for k, v in tuples:
        res.write(k.encode())
        res.write(bytes(1))
        res.write(dumps(v).encode())
        res.write(bytes(1))

    return res.getvalue()
