
NUGETS — NeUral GEomeTry Suite
------------------------------

About
=====

This is a machine learning benchmark for geometric problems.

Chores
=======
1. Clean CGAL dependencies
2. Add relative positional encodings
3. Add in graph learning utilities


Install
=======

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

** Note: Currently this uv setup will work on Linux x86_64 machines. If you are using aarch64, a conda requirements file (coming soon) will be easier for you to use **

You may need the Computational Geometry Algorithms Library (CGAL) in order to use this benchmark fully. Follow the installation directions on their documentation. You may need the following additional step (I did): 

   1. When following the CGAL installation build, modify the configuration to `cmake .. -DCMAKE=<CGAL-location>`. For me, it was `/home/CGAL-6.0`. 

   2. There may also be an error thrown about MPFR when attempting compile some of the CGAL examples. Be sure to point `CMAKE_PREFIX_PATH` to `$CONDA-PREFIX` if you are using that. 

   3. When compiling the swig bindings, you must compile it against the SAME version of python that you are running the library in. 

   With these two steps, I was able to install and use CGAL with C++17 on a linux-aarch64 machines. Eventually, I would like to move away from using these swig bindings. 

Installation for aarch64 Linux machines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing the necessary dependencies is different for aarch64 linux machines. This is because torch_scatter needs to be built from source. As of now (mid-2026), there are no pre-built official wheels for torch_scatter. 

Instead of using uv, on aarch64 machines, I have had more luck using a conda environment. A such, please start by creating a conda environment with the provided `environment.yml`. Note that these steps worked on the Vista system on the TACC cluster. If you have the chance to use an x86_64 system, I would recommend using that instead. 

.. code-block:: console

   $ conda env create -f environment.yml

This environment does not contain the necessary dependencies for torch_scatter, torch_geometric, torch_heterogeneous_batching, and lightning. 

For torch_scatter, use the guide (here)[https://github.com/rusty1s/pytorch_scatter/issues/428].

For torch_geometric: 

.. code-block:: console 

   $ pip install torch_geometric

For torch_heterogeneous_batching: 

.. code-block:: console 
   $ pip install "torch_heterogeneous_batching + git+https://github.com/chens5/pytorch_heterogeneous_batching.git@main"

Docker image for TACC
~~~~~~~~~~~~~~~~~~~~~

See `this page <https://containers-at-tacc.readthedocs.io/en/latest/singularity/03.mpi_and_gpus.html>` for more info.

To build from a standard ``x86/64``, you need to first enable cross-platform builds using qemu with this command (this only needs to be done **once**):

.. code-block:: console

   $ docker run --rm --privileged tonistiigi/binfmt --install all

Then to build the container image, just use the following:

.. code-block:: console

   $ docker buildx build -t nugets --platform linux/arm64 . --load

Or, to build using Dockerfile.slimmer

.. code-block:: console

   $ docker buildx build -t nugets-slim -f Dockerfile.slimmer --platform linux/arm64 . --load


To run it, follow `these instructions <https://containers-at-tacc.readthedocs.io/en/latest/singularity/01.singularity_basics.html>` 

Note that to run, you need to provide

* The worker id with ``--env SWEEP_ID=<the sweep id>``
* The config with ``--mount type=bind,source=path_to_the_config,target=/app/config.yaml,ro`` (i do not include it in the image because that would be terrible from a security standpoint since it has api keys.)

The simplest to get the image there should be a private docker registry (most cloud services offer that, if you have some credits), for example `in gcp the service is part of artifact registry <https://cloud.google.com/artifact-registry/docs>`. See `the doc on how to push images <https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling>`.

.. code-block:: console

Run
====
