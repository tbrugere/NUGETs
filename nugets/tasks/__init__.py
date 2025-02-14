from .task import Task
from nugets.misc import import_submodules

def get_tasks_register():
    """Get the register of all datasets

    This function imports all the datasets in the current module.
    It is gated behind a function to avoid doing that at import time.
    """
    from .register import register
    #### import all the datasets to populate the register
    import_submodules(__name__)

    return register

__all__ = ["Task", "get_tasks_register"]
