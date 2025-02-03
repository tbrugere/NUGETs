"""Geometric datasets

This module provides a set of synthetic and real-world datasets 
for evaluating neural networks on geometry tasks.
"""

import importlib
import pkgutil

def _import_submodules(package, recursive=True):
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

def get_dataset_register():
    """Get the register of all datasets

    This function imports all the datasets in the current module.
    It is gated behind a function to avoid doing that at import time.
    """
    from .register import register
    #### import all the datasets to populate the register
    _import_submodules(__name__)

    return register

