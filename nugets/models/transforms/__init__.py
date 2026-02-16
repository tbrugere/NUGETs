from .transform import PositionalEncodingTransform
from nugets.misc import import_submodules

def get_transform_register():
    """Get the register of all transforms

    This function imports all the transforms in the current module.
    It is gated behind a function to avoid doing that at import time.
    """
    from .register import register
    #### import all the datasets to populate the register
    import_submodules(__name__)

    return register

__all__ = ["PositionalEncodingTransform", "get_transform_register"]