from typing import Any, Generic, dataclass_transform, get_type_hints, overload, Literal, TypeVar, Iterator

from dataclasses import dataclass
from ml_lib.misc.data_structures import Maybe
from torch import nn

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

class IntHyperparameter(Hyperparameter[int]):
    """Integer hyperparameter"""
    def __init__(self, default: Maybe[int] = Maybe(), description: str|None = None):
        super().__init__(int, default, description)
class FloatHyperparameter(Hyperparameter[float]):
    """Float hyperparameter"""
    def __init__(self, default: Maybe[float] = Maybe(), description: str|None = None):
        super().__init__(float, default, description)

@dataclass_transform(kw_only_default=True, field_specifiers=(Hyperparameter, ))
class BackBone(nn.Module):
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
    
    def __init__(self, **kwargs):
        super().__init__()
        self.set_hyperparameters(**kwargs)
        self.__setup__()

    def __setup__(self):
        """Setup the backbone"""
        pass


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
                         provided=hyperparameters.keys,
                         missing_message=f"Missing hyperparameters for {model_name}", 
                         extra_message=f"Extra hyperparameters for {model_name}", 
                         wrong_message=f"Wrong hyperparameters for {model_name}", 
                         )
        for k, h_spec in self.list_hyperparameters(return_types=True):
            h_spec.validate(hyperparameters[k])
            setattr(self, k, hyperparameters[k])
