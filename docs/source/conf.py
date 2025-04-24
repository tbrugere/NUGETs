# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from sphinx_pyproject import SphinxConfig

import sys
sys.path.append("..")

# -- Project information -----------------------------------------------------
# we are using the pyproject.toml file to store the project metadata
# see https://github.com/sphinx-toolbox/sphinx-pyproject
config = SphinxConfig("../../pyproject.toml", globalns=globals())


templates_path = ['_templates']



