"""Miscellaneous utilities"""

from typing import Any, Callable, TypeVar, Protocol, ParamSpec
import argparse
from argparse import ArgumentParser, _ArgumentGroup
import functools as ft
from io import BytesIO
import importlib
from json import dumps as json_dumps
import pkgutil

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

class Nestedspace(argparse.Namespace):
    def __setattr__(self, name, value):
        if '.' in name:
            group,name = name.split('.',1)
            ns = getattr(self, group, Nestedspace())
            setattr(ns, name, value)
            self.__dict__[group] = ns
        else:
            self.__dict__[name] = value

class CustomArgumentParser(argparse.ArgumentParser):
    updates: list[Callable]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updates = []
        

    def add_argument_group(self, *args, prefix=None, dest_group=None, **kwargs):
        group = NamespacedArgumentGroup(self, *args, prefix=prefix, dest_group=dest_group, **kwargs)
        self._action_groups.append(group)
        return group

    def _pop_action_class(self, kwargs, default=None):
        return AddsUpdateAction.pop_action_class(self, kwargs, default)

    def parse_args(self, args=None, namespace=None):
        if namespace is None: namespace = Nestedspace()
        namespace, argv= self.parse_known_args(args, namespace)
        while argv:
            if not self.updates: 
                msg = ('unrecognized arguments: %s') % ' '.join(argv)
                if self.exit_on_error:
                    self.error(msg)
                else:
                    raise argparse.ArgumentError(None, msg)
            while self.updates:
                update = self.updates.pop()
                update(namespace)
            namespace, argv= self.parse_known_args(argv, namespace)
        return namespace

    def register_update(self, update):
        self.updates.append(update)

class AddsUpdateAction(argparse.Action):

    inner_action: argparse.Action
    update: Callable

    namespace_scope: str|None


    def __init__(self,  *args, inner_action_class, update, namespace_scope=None, **kwargs ):
        self.inner_action = inner_action_class(*args, **kwargs)
        self.update = update
        self.namespace_scope = namespace_scope

    def __call__(self, parser, namespace, values, option_string=None):
        parser.register_update(self.scoped_update)#type: ignore
        return self.inner_action(parser, namespace, values, option_string=option_string)

    def __getattr__(self, attr_name):
        return getattr(self.inner_action, attr_name)

    def __setattr__(self, attr_name, value):
        if attr_name in ["inner_action", "update", "namespace_scope"]:
            super().__setattr__(attr_name, value)
        return setattr(self.inner_action, attr_name, value)

    def scoped_update(self, namespace):
        if self.namespace_scope:
            scoped_namespace = getattr(namespace, self.namespace_scope)
        else: scoped_namespace = namespace
        return self.update(scoped_namespace)       

    @classmethod
    def pop_action_class(cls, actions_container, kwargs, default=None, namespace_scope=None,):
        action = kwargs.pop('action', default)
        update = kwargs.pop('update', None)
        inner_action = actions_container._registry_get('action', action, action)
        if update is None:
            return inner_action
        return ft.partial(cls, inner_action_class=inner_action, update=update, namespace_scope=namespace_scope)




class NamespacedArgumentGroup(_ArgumentGroup):
    """Group arguments in an argument parser in a way that

    - every element from the group is saved into a custom namespace
    - groups can be nested infinitely (yay)

    """

    def __init__(self, *args, prefix=None, dest_group=None, **kwargs):
        self.group_prefix = prefix
        self.dest_group = dest_group
        super().__init__(*args, **kwargs)
        if self.dest_group is not None:
            self.argument_default = argparse.SUPPRESS

    def _add_action(self, action):
        if self.dest_group is not None:
            action.dest = f"{self.dest_group}.{action.dest}"
        if self.group_prefix is not None:
            prefix = self.group_prefix
            old_option_strings = action.option_strings
            new_option_strings = []
            for opt_s in old_option_strings:
                dd = "--"
                assert opt_s.startswith(dd)
                body = opt_s[len(dd):]
                new_option_strings.append(f"{dd}{prefix}-{body}")

            action.option_strings = new_option_strings
        return super()._add_action(action)

    def _pop_action_class(self, kwargs, default=None):
        return AddsUpdateAction.pop_action_class(self, kwargs, default, 
                                                 namespace_scope=self.dest_group)

    def add_argument_group(self, *args, prefix=None, dest_group=None, **kwargs):
        if prefix is None: prefix = self.group_prefix
        elif self.group_prefix is None: pass
        else: prefix = f"{self.group_prefix}-{prefix}"

        if dest_group is None: dest_group = self.dest_group
        elif self.dest_group is None: pass
        else: dest_group = f"{self.dest_group}.{dest_group}"

        group = NamespacedArgumentGroup(self, *args, prefix=prefix, dest_group=dest_group, **kwargs)
        self._action_groups.append(group)
        return group


class CreateArgumentParserReturnSignature(Protocol): #noqa: D101
    def __call__(self, parser: CustomArgumentParser|None = None) -> CustomArgumentParser: ... #noqa: D102

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
        def wrapper(parser: CustomArgumentParser|None = None) -> CustomArgumentParser:
            if parser is None:
                parser = CustomArgumentParser(**argparse_kwargs)
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

def import_submodules(package, recursive=True):
    """Import all submodules of a module, recursively, including subpackages"""
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        try:
            results[full_name] = importlib.import_module(full_name)
        except ModuleNotFoundError:
            continue
        if recursive and is_pkg:
            results.update(_import_submodules(full_name))
    return results

