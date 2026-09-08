NUGETS — NeUral GEomeTry Suite
================================

About
-----

This is a machine learning benchmark for geometric problems.

Chores (ordered from most to least urgent)
-----------------------------------------

1. Update documentation

   A. Executing single runs and hyperparameter searches.
   B. Installing CGAL from a personal SWIG fork.

2. Add in graph learning utilities
3. Add relative positional encodings
4. Clean CGAL dependencies (eventually, I would like to move away from CGAL entirely.)
5. Add in more sophisticated sampling options (Morse-based sampling on NZDEM dataset.)

Install
-------

Local
~~~~~

This project uses `uv <https://github.com/astral-sh/uv>`_ to manage dependencies.

To install the dependencies, make sure ``uv`` is installed and run:

.. code-block:: console

   $ uv sync
   $ # if you need development dependencies, use this instead:
   $ uv sync --all-groups


It will automatically create a virtual environment in ``.venv``, which you can then use by running:

.. code-block:: console

   $ source .venv/bin/activate

.. note::

   The uv configuration targets Linux x86_64 with Python 3.11.
   For aarch64, see the installation notes below.

This virtual environment is all that is needed to start experimenting with distance tasks, convex hulls, geometric primitives such as range searching, and epsilon-kernels.

CGAL installation for shapefitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For certain tasks, a Python wrapper for the Computational Geometry Algorithms Library (CGAL) is necessary. These are the shapefitting tasks and the alpha shape task. In order for the Python wrapper to compile, you will also need to have CGAL available. Follow the directions `on the CGAL download page <https://www.cgal.org/download.html>`_.

Once you have CGAL installed, follow the installation directions in the wiki of `this repository <https://github.com/chens5/cgal-swig-bindings>`_. Do not install the python package via pip. You may need the following additional steps:

   1. When following the CGAL installation build, modify the configuration to ``cmake .. -DCGAL_DIR=<CGAL-config-directory>``. Use the directory containing ``CGALConfig.cmake``.

   2. There may also be an error thrown about MPFR when attempting compile some of the CGAL examples. Be sure to point ``CMAKE_PREFIX_PATH`` to ``$CONDA_PREFIX`` if you are using that.

   3. When compiling the swig bindings, you must compile it against the SAME version of python that you are running the library in.


Installation for aarch64 Linux machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing the necessary dependencies is different for aarch64 linux machines. This is because torch_scatter needs to be built from source. The wheel pinned in ``pyproject.toml`` targets Linux x86_64.

An aarch64 conda ``environment.yml`` file is not currently included in this
repository. 
Note that these steps worked on the Vista system on the TACC cluster. In general, x86_64 systems are recommended if at all possible.

.. code-block:: console

   $ conda env create -f environment.yml

This environment does not contain the necessary dependencies for torch_scatter, torch_geometric, torch_heterogeneous_batching, and lightning.

For torch_scatter, use the guide `here <https://github.com/rusty1s/pytorch_scatter/issues/428>`_.

For torch_geometric:

.. code-block:: console

   $ pip install torch_geometric

For torch_heterogeneous_batching:

.. code-block:: console

   $ pip install "torch-heterogeneous-batching @ git+https://github.com/chens5/pytorch_heterogeneous_batching.git@main"

.. Docker image for TACC
.. ~~~~~~~~~~~~~~~~~~~~~

.. See `this page <https://containers-at-tacc.readthedocs.io/en/latest/singularity/03.mpi_and_gpus.html>` for more info.

.. To build from a standard ``x86/64``, you need to first enable cross-platform builds using qemu with this command (this only needs to be done **once**):

.. .. code-block:: console

..    $ docker run --rm --privileged tonistiigi/binfmt --install all

.. Then to build the container image, just use the following:

.. .. code-block:: console

..    $ docker buildx build -t nugets --platform linux/arm64 . --load

.. Or, to build using Dockerfile.slimmer

.. .. code-block:: console

..    $ docker buildx build -t nugets-slim -f Dockerfile.slimmer --platform linux/arm64 . --load


.. To run it, follow `these instructions <https://containers-at-tacc.readthedocs.io/en/latest/singularity/01.singularity_basics.html>`

.. Note that to run, you need to provide

.. * The worker id with ``--env SWEEP_ID=<the sweep id>``
.. * The config with ``--mount type=bind,source=path_to_the_config,target=/app/config.yaml,ro`` (i do not include it in the image because that would be terrible from a security standpoint since it has api keys.)

.. The simplest to get the image there should be a private docker registry (most cloud services offer that, if you have some credits), for example `in gcp the service is part of artifact registry <https://cloud.google.com/artifact-registry/docs>`. See `the doc on how to push images <https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling>`.

.. .. code-block:: console

Run
---

This project relies heavily on wandb for tracking metrics as well as hyperparameter searching. As such, make a ``config.yaml`` file which contains your wandb API key and the name of the wandb_project which you would like to log to. Format it as follows:

.. code-block:: yaml

   wandb_key: <your-api-key>
   wandb_project: NUGETS


In order to run a single configuration, simply run the following command:

.. code-block:: console

   $ python -m nugets train_from_config experiment-config.yaml --n-epochs 100

There are several experiment configurations available under ``static_configs/single_runs``.

More information regarding running the code and the structure of this repository can be found in the documentation.
