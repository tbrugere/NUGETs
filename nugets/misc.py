"""Miscellaneous utilities"""

from typing import Any, Callable, TypeVar, Protocol, ParamSpec
import argparse
from argparse import ArgumentParser, _ArgumentGroup
from gettext import gettext as _
import functools as ft
from io import BytesIO
import importlib
from inspect import signature
from json import dumps as json_dumps
import pkgutil
from logging import getLogger
import logging
from pathlib import Path
import sys

log = getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

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

    def __getattr__(self, name):
        if '.' in name:
            group,name = name.split('.',1)
            ns = getattr(self, group, Nestedspace())
            return getattr(ns, name)
        else:
            return getattr(super(), name)


class CustomHelpAction(argparse._HelpAction):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.should_print_help = True

class RetryException(Exception):
    pass

class CustomArgumentParser(argparse.ArgumentParser):
    """
    Known bugs: 
        - if using --help, and some required arguments are missing at a higher level, not all updates will be processed, so not all possible arguments will be shown
        Solution to that would be to prevent exiting on unseen argument if there are still updates to be taken, but that is for sure a pain

    """
    updates: list[Callable]

    def __init__(self, *args, **kwargs):
        bound_args = signature(argparse.ArgumentParser).bind(*args, **kwargs)
        bound_args.apply_defaults()
        add_help = bound_args.arguments["add_help"]
        bound_args.arguments["add_help"] = False
        super().__init__(*bound_args.args, **bound_args.kwargs)
        self.register('action', 'help', CustomHelpAction)
        if add_help:
            prefix_chars = self.prefix_chars
            default_prefix = '-' if '-' in prefix_chars else prefix_chars[0]
            self.add_argument(
                default_prefix+'h', default_prefix*2+'help',
                action='help', default=argparse.SUPPRESS,
                help=_('show this help message and exit'))

        self.should_print_help = False
        self.updates = []
        self._retry_on_error = False
        

    def add_argument_group(self, *args, prefix=None, dest_group=None, **kwargs):
        group = NamespacedArgumentGroup(self, *args, prefix=prefix, dest_group=dest_group, **kwargs)
        self._action_groups.append(group)
        return group

    def add_subparsers(self, **kwargs):
        if "parser_class" not in kwargs:
            kwargs["parser_class"] = self.__class__
        return super().add_subparsers(**kwargs)

    def _pop_action_class(self, kwargs, default=None):
        return AddsUpdateAction.pop_action_class(self, kwargs, default)

    def parse_args(self, args=None, namespace=None):
        try:
            return super().parse_args(args, namespace)
        except argparse.ArgumentError:
            self._eventually_print_help(self._current_namespace)
            raise

    def _eventually_print_help(self, namespace):
        # called in a situation when about to finish parsing. 
        # (whether cleanly or on error)
        # If it was asked by arguments to print help, do that instead of returning/failing
        if not self.should_print_help: return
        self.run_updates(namespace)
        self.print_help()
        self.exit()

    def error(self, message, dontretry=False):
        if getattr(self, "_retry_on_error", False) and not dontretry:
            self._error_message = message
            raise RetryException
        self._eventually_print_help(self._current_namespace)
        return super().error(message)

    def previous_error(self):
        previous_error = getattr(self, "_error_message", None)
        if previous_error is not None:
            self.error(previous_error, dontretry=True)
        

    def parse_known_args(self, args=None, namespace=None):
        if namespace is None: namespace = Nestedspace()
        self._retry_on_error = True
        self._current_namespace = namespace # kinda hacky, making the class more stateful
        # needed for accessing the namespace on error
        self.should_print_help = False
        try:
            self._error_message = None
            namespace, unconsumed_args = super().parse_known_args(args, namespace)
        except RetryException:
            unconsumed_args = args
        while unconsumed_args:
            ran_updates = self.run_updates(namespace)
            if not ran_updates: 
                self.previous_error()
                return namespace, unconsumed_args
            self._error_message = None
            namespace, unconsumed_args = super().parse_known_args(args, namespace)

        self._eventually_print_help(namespace)
        return namespace, unconsumed_args

    # def _parse_known_args2(self, args, namespace, intermixed):
    #     if namespace is None: namespace = Nestedspace()
    #     self._current_namespace = namespace # kinda hacky, making the class more stateful
    #     return super()._parse_known_args2(args, namespace, intermixed)

    def format_help(self):
        formatter = self._get_formatter()

        # usage
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)

        # description
        formatter.add_text(self.description)

        def format_group(action_group):
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            for subgroup in action_group._action_groups:
                format_group(subgroup)
            formatter.end_section()

        # positionals, optionals and user-defined groups
        for action_group in self._action_groups:
            format_group(action_group)

        # epilog
        formatter.add_text(self.epilog)

        # determine help from format above
        return formatter.format_help()



    def run_updates(self, namespace):
        """Runs all pending updates.
        Returns:
            True if any update was run, False otherwise
        """
        if not self.updates: return False
        while self.updates:
            update = self.updates.pop()
            update(namespace)
        return True


    def register_update(self, update):
        self.updates.append(update)



class AddsUpdateAction(argparse.Action):
    inner_action: argparse.Action
    update: Callable
    already_ran: bool
    already_registered: bool

    namespace_scope: str|None


    def __init__(self,  *args, inner_action_class, update, namespace_scope=None, **kwargs ):
        self.inner_action = inner_action_class(*args, **kwargs)
        self.update = update
        self.namespace_scope = namespace_scope
        self.already_ran = False
        self.already_registered = False

    def __call__(self, parser, namespace, values, option_string=None):
        if not self.already_registered:
            parser.register_update(self.scoped_update)#type: ignore
            self.already_registered=True
        return self.inner_action(parser, namespace, values, option_string=option_string)

    def __getattr__(self, attr_name):
        return getattr(self.inner_action, attr_name)

    def __setattr__(self, attr_name, value):
        if attr_name in ["inner_action", "update", "namespace_scope"]:
            super().__setattr__(attr_name, value)
        return setattr(self.inner_action, attr_name, value)

    def scoped_update(self, namespace):
        if self.already_ran: log.warn("update ran twice, this shouldn't happen")
        self.already_ran = True
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


def configure_logging(packages, loglevel, logfile=None): 
    """Configures the logging for the program

    Enables logging (either to stderr or to a file) for the following packages:

    - autocommit
    - basic_rag
    - mistral_tools

    with the specified log level

    Args:
        loglevel (int): The log level
        logfile (Path, optional): The file to log to. Defaults to None.
    """
    if logfile is not None:
        logfile = Path(logfile)
        logfile.parent.mkdir(exist_ok=True, parents=True)
        if loglevel > logging.INFO:
            loglevel = logging.INFO
    log.setLevel(loglevel)
    formatter = logging.Formatter(
            '%(asctime)s:%(levelname)s:%(name)s:%(message)s', 
            datefmt='%H:%M:%S')
    if logfile is not None:
        handler = logging.FileHandler(logfile)
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    for package in packages:
        our_log = getLogger(package)
        our_log.addHandler(handler)


def compact_wandb_sweep_config(config_dict):
    # unfinished. do not use

    non_recursive_parameter_keywords = {"values", "value", "distribution", "probabilities", "min", "max", "mu", "sigma", "q"}
    def is_nonrecursive_parameter(d: dict):
        # a bit simplistic but 
        return all(key in non_recursive_parameter_keywords for key in d)

    def deserialize(parameter_object, inside_parameter=False):
        # inside_parameter: whether this is a key-value dict representing parameters (as opposed to inside of a parameter)


        match parameter_object:
            case dict() if not inside_parameter:
                return {key: deserialize(value, inside_parameter=True) 
                        for key, value in parameter_object}
            case _ if not inside_parameter:
                raise ValueError("toplevel for parameters should be a dict")
            case dict() if is_nonrecursive_parameter(parameter_object):
                return parameter_object
            # case à

