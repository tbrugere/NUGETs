
NUGETS — NeUral GEomeTry Suite
------------------------------


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

Docker image for TACC
~~~~~~~~~~~~~~~~~~~~~

See `this page <https://containers-at-tacc.readthedocs.io/en/latest/singularity/03.mpi_and_gpus.html>` for more info.

To build from a standard ``x86/64``, you need to first enable cross-platform builds using qemu with this command (this only needs to be done **once**):

.. code-block:: console

   $ docker run --rm --privileged tonistiigi/binfmt --install all

Then to build the container image, just use the following:

.. code-block:: console

   $ docker buildx build -t nugets --platform linux/arm64 .

Or, to build using Dockerfile.slimmer

.. code-block:: console

   $ docker buildx build -t nugets-slim -f Dockerfile.slimmer --platform linux/arm64 .

To run it, follow `these instructions <https://containers-at-tacc.readthedocs.io/en/latest/singularity/01.singularity_basics.html>` 

The simplest to get the image there should be a private docker registry (most cloud services offer that, if you have some credits), for example `in gcp the service is part of artifact registry <https://cloud.google.com/artifact-registry/docs>`. See `the doc on how to push images <https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling>`.

.. code-block:: console

Run
====
