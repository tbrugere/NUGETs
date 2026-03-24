NUGETs architecture overview
=========================

The NUGETs package is meant to allow people to easily run their own models to investigate approximating solutions to computational geometric problems with neural networks. 

Workflow
-----------------
The NUGETs package is separated out into three major parts: 
#. Datasets: this is the raw point cloud data.
#. Task: maps each dataset to a label, depending on the problem you are interested in. 
#. Model: model which is used to approximate solutions. We follow the Encode-Process-Decode paradigm of (Hamrick). 

For example, if I am interested in approximating the minimum enclosing ball for the ModelNet dataset, we start with the raw point cloud data representing each object in ModelNet. Then, it gets mapped into the following training point (S, c, r) where c and r are the center and radius of the minimum enclosing ball, respectively. The model the learns to approximate c and r. 

Using your own datasets
-----------------
It is easy to add your own datasets as well. 

To add 

I added a ``nugets.sh`` script that essentially 

#. sets stuff in the the environment
#. runs the code *with automated debugging* (meaning the ``ipdb`` debugger will automatically run if an exception is not caught)

.. note::

   This is not fully optimal in a production environment:

   * running with automated debugging may have a small performance overhead (not sure this matters when running on gpu though)
   * running this way makes ``torch-lightning`` think it is in an "interactive environment" (such as a jupyter notebook) because I am using ``ipython`` there. This makes it disable some features such as multi-gpu training (although I don't think we want to enable that)


.. code-block:: console

   $ ./nugets.sh  --help
   usage: __main__.py [-h] [--config CONFIG] [-d] [-v] [--logfile LOGFILE]
                   {train,train_from_config,wandb_agent} ...

   NUGETS - NeUral GEomeTry Suite

   positional arguments: {train,train_from_config,wandb_agent}
       train               Train a model
       train_from_config   train a model from a config file
       wandb_agent         run the agent for a wandb sweep

   options:
     -h, --help            show this help message and exit
     --config CONFIG       global config path
     -d, --debug           Print lots of debugging statements
     -v, --verbose         Be verbose
     --logfile LOGFILE     Log file to write to (implies --verbose)


Working directory
-----------------

I generally store every untracked file in an untracked "working directory" called ``workdir``, at the root of the project. 
It should live on a data drive (ie one with high capacity). On ``ds-serv2``, I make it a softlink to a subdir of ``/data``.

Here is its structure:

.. code-block:: bash

   workdir
   ├── datasets
   │   ├── configs # contains dataset configurations. consider moving this to a tracked directory
   │   ├── processed # contains processed datasets
   │   └── raw # contains raw datasets
   └── models # contains checkpoints and training results

all of these directories will be automatically created when the code is run (except for ``workdir/datasets/configs``. There is no reference in the code to ``workdir/datasets/configs`` either, so you can just move it wherever)

.. _config-file:

Config file
-----------

The program will read some values from a config file at the root of the project called ``config.yaml``.

Here is what I use (for the rest I use the default)

.. code-block:: yaml

   wandb_key: [the api key for weights and biases]
   wandb_project: [the project name for weights and biases]
   processed_dataset_bucket: [the gcp bucket containing processed datasets]
   checkpoint_bucket: [the gcp bucket containing model checkpoints and training run info]

The full definition of all fields for the config file is in :py:class:`nugets.pipeline.config.GlobalConf`.


Controlling which GPU is being used
-----------------------------------


By default the ``cuda:0`` device will be used, but another one can be selected.
This is done by setting the ``CUDA_VISIBLE_DEVICES`` environment variable in the shell.

e.g.

.. code-block:: console

   $ export CUDA_VISIBLE_DEVICES=2
   $ ./nugets.sh [...] # should use GPU 2

Training
--------


Using only the command line (not recommanded)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the ``train`` subcommand to specify all model parameters throguh the command line.
I don't recommend this because it leads to very long commands, this is mostly a result of me having fun with argparse, but it can be useful for quick testing (especially for models with not too many hyperparameters or with sensible defaults)

There is an example is an example for that in the ``test_train_command.sh file``. The command looks like this:

.. code-block:: console

   .venv/bin/python -m nugets train \
       --batch-size 12 --learning-rate 1.e-6 \
       --task "WassersteinDistanceTask" \
       --dataset workdir/datasets/configs/dummy_growing_circles.yaml \
       --backbone-type CoupledNetwork  \
       --backbone-latent-dimension 256 \
       --backbone-p 2 \
       --backbone-decoder-distance SinkhornLoss \
       --backbone-encoder-backbone Transformer \
           --backbone-encoder-n-heads 12 \
           --backbone-encoder-n-layers 6 \
           --backbone-encoder-d-model 768 \
           --backbone-encoder-feed-forward-hidden-dim 51

.. note::

   Here the dataset is still specified through a yaml config.

.. _training-run-config:

Using config files
~~~~~~~~~~~~~~~~~~

.. note::

   This section is about model config files. For the global config file see

You can use the ``train_from_config`` subcommand to use a config file to specify training parameters from a YAML file instead. A good location to store those could be ``workdir/configs`` for ex.

.. code-block:: console

   .venv/bin/python -m nugets train_from_config workdir/model_config.yaml

where ``model_config.yaml`` has the following structure


.. code-block:: yaml

   task:
      type: TaskName
      dataset: DatasetName
      dataset_config:
         dataset_parameter1: value1
         dataset_parameter2: value2
   backbone:
      type: BackboneName
      hyperparameter1: value1
      hyperparameter2: value2
      [...]
   batch_size: 16
   learning_rate: 1.e-5
   debug_mode: False

Using Weights and biases sweeps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See also: :doc:`cloud_integration`.

This is the solution for automation: you can run a W&B sweep worker by using the following command:

.. code-block::

   ./nugets.sh wandb_agent $SWEEP_ID
