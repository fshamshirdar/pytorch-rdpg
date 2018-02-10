======
Recurrent Deterministic Policy Gradient (RDPG)
======

Overview
======
`PyTorch <https://github.com/pytorch/pytorch>`_ implementation of Recurrent Deterministic Policy Gradient from the paper `Memory-based control with recurrent neural networks <https://arxiv.org/abs/1512.04455>`_ 

Run
======
* Training : results of two environment and their training curves:

	* Pendulum-v0

	.. code-block:: console

	    $ ./main.py --debug

	.. image:: output/Pendulum-v0-run0/validate_reward.png
	    :width: 800px
	    :align: left
	    :height: 600px
	    :alt: alternate text

	* MountainCarContinuous-v0

	.. code-block:: console

	    $ ./main.py --env MountainCarContinuous-v0 --validate_episodes 100 --max_episode_length 2500 --ou_sigma 0.5 --debug

	.. image:: output/MountainCarContinuous-v0-run0/validate_reward.png
	    :width: 800px
	    :align: left
	    :height: 600px
	    :alt: alternate text

References: 
======
`Memory-based control with recurrent neural networks <https://arxiv.org/abs/1512.04455>`_
`Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>`_
`DDPG implementation using PyTorch <https://github.com/ghliu/pytorch-ddpg>`_
`PyTorch-RL <https://github.com/jingweiz/pytorch-rl>`_
