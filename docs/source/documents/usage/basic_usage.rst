Quick Start
=======================

.. raw:: html

   <br><hr>

Run a DRL example
-----------------------

In XuanCe, it is easy to build a DRL agent. First you need to create a *runner*
and specify the ``agent_name``, ``env_name``, then a runner that contains agent, policy, and envs, etc., will be built. 
Finally, execute ``runner.run`` and the agent's model is training.

.. code-block:: python

    import xuance
    runner = xuance.get_runner(method='dqn',
                               env='classic_control',
                               env_id='CartPole-v1',
                               is_test=False)
    runner.run()

After training the agent, you can test and view the model by the following codes:

.. raw:: html

   <br><hr>

Run an MARL example
-----------------------

XuanCe support MARL algorithms with both cooperative and competitive tasks.
Similaly, you can start by:

.. code-block:: python

    import xuance
    runner = xuance.get_runner(method='maddpg',
                               env='mpe',
                               env_id='simple_spread_v3',
                               is_test=False)
    runner.run()

For competitve tasks in which agents can be divided to two or more sides, you can run a demo by:

.. code-block:: python

    import xuance
    runner = xuance.get_runner(method=["maddpg", "iddpg"],
                               env='mpe',
                               env_id='simple_push_v3',
                               is_test=False)
    runner.run()

In this demo, the agents in `mpe/simple_push <https://pettingzoo.farama.org/environments/mpe/simple_push/>`_ environment are divided into two sides, named "adversary_0" and "agent_0".
The "adversary"s are MADDPG agents, and the "agent"s are IDDPG agents.

Test
-----------------------

After completing the algorithm training, XuanCe will save the model files and training log information in the designated directory.
Users can specify "is_test=True" to perform testing.

.. code-block:: python

    import xuance
    runner = xuance.get_runner(method='dqn',
                               env_name='classic_control',
                               env_id='CartPole-v1',
                               is_test=True)
    runner.run()

In the above code, "runner.benchmark()" can also be used instead of "runner.run()" to train benchmark models and obtain benchmark test results.

Logger
-----------------------

You can use the tensorboard or wandb to visualize the training process by specifying the "logger" parameter in the "xuance/configs/basic.yaml".

.. code-block:: yaml

    logger: tensorboard

or

.. code-block:: yaml

    logger: wandb

**1. Tensorboard**

After completing the model training, the log files are stored in the "log" folder in the root directory.
The specific path depends on the user's actual configuration.
Taking the path "./logs/dqn/torch/CartPole-v0" as an example, users can visualize the logs using the following command:

.. code-block:: bash

    tensorboard --logdir ./logs/dqn/torch/CartPole-v1/

**2. W&B**

If you choose to use the wandb tool for training visualization,
you can create an account according to the official W&B instructions and specify the username "wandb_user_name" in the "xuance/configs/basic.yaml" file.

For information on using W&B and its local deployment, you can refer to the following link:

| **wandb**: `https://github.com/wandb/wandb.git <https://github.com/wandb/wandb.git/>`_
| **wandb server**: `https://github.com/wandb/server.git <https://github.com/wandb/server.git/>`_