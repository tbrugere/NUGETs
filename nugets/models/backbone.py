from typing import Any, Generic, dataclass_transform, get_type_hints, overload, Literal, TypeVar, Iterator, TYPE_CHECKING, ClassVar, Self

from dataclasses import dataclass
from logging import getLogger

from ml_lib.misc.data_structures import Maybe, SingletonMeta
import torch
from torch import nn

from nugets.pipeline.configs import BackboneConf

log = getLogger(__name__)

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace


T = TypeVar('T')

@dataclass
class Hyperparameter(Generic[T]):
    """Hyperparameters specification for a backbone"""
    type: type[T]
    default: Maybe[T] = Maybe()
    description: str|None = None

    def validate(self, val: T) -> None:
        """Validate the hyperparameter value"""
        if not isinstance(val, self.type):
            raise ValueError(f"Expected {self.type}, got {type(val)}")

    def serialize(self, val: str|dict) -> T:
        """Parse the hyperparameter value from a config"""
        return self.type(val)

    def get_dest(self, attr_name):
        return attr_name

    def add_argparse_arguments(self, group, attr_name):
        attr_name_arg = attr_name.replace("_", "-")
        argument = f"--{attr_name_arg}"

        optional = not self.default.is_empty
        dest = self.get_dest(attr_name)

        if optional:
            group.add_argument(argument, type=self.type,
                                dest=dest, default=self.default.value,
                                required=False,
                                help=self.description)
        else:
            group.add_argument(argument, type=self.type,
                            dest=dest, required=True,
                            help=self.description)

    def read_argparse_arguments(self, namespace, attr_name):
        dest = self.get_dest(attr_name)
        val = getattr(namespace, dest)
        return val

@dataclass
class InnerBackbone():
    t: "type[BackBone]"
    params: "Namespace|dict"

    def load(self):
        if isinstance(self.params, dict):
            return self.t.from_dict(self.params)
        return self.t.from_args(self.params)


class OtherBackboneHyperparameter(Hyperparameter[InnerBackbone]):
    """Hyperparameter that loads parameters for another backbone"""
    type = InnerBackbone
    description: str|None = None

    def __init__(self, description: str|None = None):
        self.type = InnerBackbone
        self.description = description

    def serialize(self, val:dict):
        config = BackboneConf.model_validate(val)
        return InnerBackbone(config.get_type(), config.get_parameters())
    
    def validate(self, val: InnerBackbone):
        pass # for now
        # val.t.check_argparse_parameters(val.params)

    def add_argparse_arguments(self, group, attr_name):
        from nugets.models.backbones import get_backbones_register
        register = get_backbones_register()
        attr_name_arg = attr_name.replace("_", "-")

        dest_group = self.get_dest(attr_name)
        subgroup = group.add_argument_group(title=f"{attr_name} parameters", 
                            prefix=attr_name_arg, dest_group=dest_group)


        def add_backbone_parameters(args):
            backbone_name = args.backbone
            backbone_type = register[backbone_name]
            backbone_type.argument_parser(subgroup)


        subgroup.add_argument("--backbone", type=str, 
                              choices=list(register),
                              update=add_backbone_parameters, 
                              required=True)


    def read_argparse_arguments(self, namespace, attr_name):
        from nugets.models.backbones import get_backbones_register
        register = get_backbones_register()
        dest = self.get_dest(attr_name)
        sub_namespace = getattr(namespace, dest)
        backbone_name = sub_namespace.backbone
        backbone_type = register[backbone_name]
        return InnerBackbone(backbone_type, sub_namespace)


class Unspecified(metaclass=SingletonMeta):
    pass

def hyperparameter(type, default=Maybe(), description: str|None = None) -> Any:
    return Hyperparameter(type, default=default, description=description)
def int_hyperparameter(default=Maybe(), description: str|None = None) -> int:
    return hyperparameter(int, default=default, description=description)
def float_hyperparameter(default=Maybe(), description: str|None = None) -> float:
    return hyperparameter(float, default=default, description=description)
def other_backbone_hyperparameter(description: str|None = None) -> InnerBackbone:
    return OtherBackboneHyperparameter(description=description)#type: ignore

def model_attribute(*, init=False) -> Any:
    """used to specify that a model attribute is not a hyperparameter"""
    del init
    return Unspecified()

class BackBoneMeta(type):
    def __new__(cls, name, bases, namespace, **kwds):
        assert "_required_model_attributes" not in namespace
        namespace_content = list(namespace.items())
        _required_model_attributes = set()
        for k, v in namespace_content:
            if v is Unspecified(): 
                del namespace[k]
                _required_model_attributes.add(k)
        namespace["_required_model_attributes"] = _required_model_attributes
        return super(BackBoneMeta, cls).__new__(cls, name, bases, namespace, **kwds)
            

@dataclass_transform(kw_only_default=True, field_specifiers=(hyperparameter, int_hyperparameter, float_hyperparameter, other_backbone_hyperparameter))
class BackBone(nn.Module, metaclass=BackBoneMeta):
    """Base class for backbone models

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument('--model_name', type=str, default='gan', help='gan or mnist')

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()
    Backbones have:

        - a set of hyperparameters
        - learnable parameters
    """

    _required_model_attributes: ClassVar[set[str]]
    
    def __init__(self, **kwargs ):
        super().__init__()
        # self._model_attributes = model_attributes
        default_attributes = set(vars(self))
        self.set_hyperparameters(kwargs)
        self.__setup__()
        self.check_initialized(default_attributes)

    def __setup__(self):
        """Setup the backbone"""
        pass

    def check_initialized(self, default_attributes):
        hyperparams = set(self.list_hyperparameters())
        for k in self._required_model_attributes:
            if k in vars(self):
                raise ValueError(f"{self.__class__.__name__}: attribute {k} was not in __setup__")
        for k in vars(self):
            # if k == "transformer": import pdb;pdb.set_trace()
            if k in hyperparams: continue
            if k in self._required_model_attributes: continue
            if k in default_attributes : continue
            log.warn(f"{self.__class__.__name__}: attribute {k} was set in __setup__"
                         "but not declared in type hints")



    @overload
    def forward(self, encoded: Any, return_reg_loss: Literal[True]) \
            -> tuple[Any, torch.Tensor|None]:
        ...

    @overload
    def forward(self, encoded: Any, return_reg_loss: Literal[False] = False) \
            -> tuple[Any, None]:
        ...

    def forward(self, encoded: Any, return_reg_loss: bool = False) \
            -> tuple[Any, torch.Tensor|None]:
        """Forward pass of the backbone.

        May return a regularization loss.

        Args:
            encoded: The output of the encoder
            return_reg_loss: Whether to return a regularization loss

        Returns:
            The output of the backbone
            The regularization loss, if requested and if applicable
        """
        del encoded, return_reg_loss
        raise NotImplementedError

    def get_input_dim(self):
        """Get the input dimension of the backbone"""
        raise NotImplementedError

    def get_output_dim(self):
        """Get the output dimension of the backbone"""
        raise NotImplementedError

    """
    Hyperparameters
    ---------------
    """

    @overload
    @classmethod
    def list_hyperparameters(cls, return_types: Literal[False] = False)\
            -> Iterator[str]:
        ...

    @overload
    @classmethod
    def list_hyperparameters(cls, return_types: Literal[True])\
            -> Iterator[tuple[str, Hyperparameter]]:
        ...


    @classmethod
    def list_hyperparameters(cls, return_types: bool = False): # type: ignore
        """List the hyperparameters of the backbone"""
        for attr_name, _ in get_type_hints(cls).items():
            val = getattr(cls, attr_name, None)
            if not isinstance(val, Hyperparameter): 
                continue
            if return_types: yield (attr_name, val)
            else: yield attr_name

    @classmethod
    def fill_hyperparameters_with_defaults(cls, hyperparameters: dict[str, Any])\
            -> dict[str, Any]:
        """Fill the hyperparameters with default values"""
        hp_with_defaults = {**hyperparameters}
        for attr_name, t in cls.list_hyperparameters(return_types=True):
            if attr_name in hyperparameters: continue
            if t.default.is_empty: continue
            hp_with_defaults[attr_name] = t.default
        return hp_with_defaults

    def set_hyperparameters(self, kwargs):
        """Set the hyperparameters of the backbone"""
        from ml_lib.misc.data_structures import check_parameters
        hyperparameters = self.fill_hyperparameters_with_defaults(kwargs)
        model_name = self.__class__.__name__
        check_parameters(required=self.list_hyperparameters(), 
                         provided=hyperparameters.keys(),
                         missing_message=f"Missing hyperparameters for {model_name}", 
                         extra_message=f"Extra hyperparameters for {model_name}", 
                         wrong_message=f"Wrong hyperparameters for {model_name}", 
                         )
        for k, h_spec in self.list_hyperparameters(return_types=True):
            h_spec.validate(hyperparameters[k])
            setattr(self, k, hyperparameters[k])

    @classmethod
    def argument_parser(cls, parser,):
        """Add hyperparameters to the argument parser"""
        for attr_name, t in cls.list_hyperparameters(return_types=True):
            t.add_argparse_arguments(parser, attr_name)
        return parser
    
    @classmethod
    def from_args(cls, args: "Namespace") -> Self:
        """Create a backbone from arguments"""
        kwargs = {}
        for attr_name, t in cls.list_hyperparameters(return_types=True):
            kwargs[attr_name] = t.read_argparse_arguments(args, attr_name)
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, args: dict) -> Self:
        """Create a backbone from arguments"""
        kwargs = {}
        for attr_name, t in cls.list_hyperparameters(return_types=True):
            kwargs[attr_name] = t.serialize(args[attr_name])
        return cls(**kwargs)
