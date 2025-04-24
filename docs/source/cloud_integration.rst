Cloud integration and automatisation
====================================

GCP
---

GCP is used for two things

* Storing model weights and training info
* Storing pre-processed datasets (to avoid re-processing them)

This can be setup as follows

1. Connect to GCP using the GCP command line. This will store credentials in a GCP-specific location, and then nugets will be able to use them. If you are running as normal user, the command is 

   .. code-block::
      
      $ gcloud auth

   If running on a google cloud machine, there is an automated way to do the login. The command is :
   
   .. code-block::

      $ gcloud activate-service-account

2. Store appropriate bucket names in the config file (see :ref:`config-file`)


Weights and biases
------------------

To use the weights and biases integration, you need to fill out three variables in the config file (see :ref:`config-file`).

.. code-block:: yaml

   wandb_key: [the api key for weights and biases]
   wandb_project: [the project name for weights and biases]

Once that is done, the project will use W&B for 2 things:

#. logging training runs.
#. sweeps (see next paragraph)


Hyperparameter sweeps
~~~~~~~~~~~~~~~~~~~~~

They are run through weights and biases, by using the `W&B Sweep API <https://docs.wandb.ai/guides/sweeps/>`_

Sweeps in weights and biases rely on two things
(I would recommend using ``yaml`` and cli, in a tracked git directory, it looks easier to keep track of)

#. a `config <https://docs.wandb.ai/guides/sweeps/define-sweep-configuration/>`_ for the sweep (that is passed to the agent). Note that we use nested parameters, the configuration structure for the hyperparameters is the same one as the one in :ref:`training-run-config`
#. an controller that orchestrates the sweeps. This can be the W&B cloud, and it can be run through the ``wandb sweep`` command line (see `here <https://docs.wandb.ai/guides/sweeps/initialize-sweeps/>`_)
#. workers (also called agents), which are processes that are running on the actual machines with GPUs. The workers get info from the agent on the actual training passes to run, and run them.


Workers can be started with

.. code-block:: console

   ./nugets.sh wandb_agent $SWEEP_ID


where ``$SWEEP_ID`` is the sweep id returned by the ``wandb sweep`` cli.

