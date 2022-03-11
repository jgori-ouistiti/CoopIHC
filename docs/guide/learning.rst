.. learning:

Using Reinforcement Learning
===============================



The main structure of *CoopIHC* is a multi-agent decision making model known as a :ref:`Partially Observable Stochastic Game<Decision-Theoretic Models>`. Because of that, it is relatively easy to convert a CoopIHC ``Bundle`` to other decision-making models. In this example, we cover a transformation to a single-agent decision-making model---known as the Partially Observable Markov Decision Process (POMDP)--- which we attempt to solve with an off-the-shelf, model-free, Deep Reinforcement Learning (DRL) algorithm. This will give us a trained policy, which we can then further use as any other policy for an agent.

The steps of this example are:

1. Define the Bundle as usual, where the policy of the agent to be trained has the right actions (without a mechanism to select those actions).
2. Wrap the bundle in a ``TrainGym`` wrapper, making it compatible with the `Gym API <https://gym.openai.com/>`_ --- a widely used standard in DRL research.
3. Train the agent's policy (i.e. attempt to solve the underlying POMDP). To do so, a machinery entirely specific to DRL and not *CoopIHC* is used. We will use `Stable-Baselines 3 (SB3) <https://stable-baselines3.readthedocs.io/en/master/>`_ to do so, which may add a few more constraints for step 2.
4. Apply the proper wrapper to make the trained policy compatible with *CoopIHC*.

A graphical representation of these steps is shown below.

.. figure:: images/training_drl.png
    :width: 800
    
    A complete workflow using Deep RL, where a Bundle is wrapped as a Gym environment, an off-the-shelf learning algorithm is used, and the trained model is wrapped as a policy to be used in CoopIHC.


.. note::

    You are not obliged to use SB3 nor Gym at all, but you may have to code your own set of wrappers if you choose not to do so. The existing code should however be relatively easy to adapt to accommodate other libraries.


Defining the Bundle
-----------------------

We use predefined objects of the pointing problem in *CoopIHC-Zoo*. The goal  in this example is to formulate a user model that is close to real human behavior. To do so, we assume that human behavior is optimal and will seek to maximize rewards. As a result, we obtain the human-like policy by solving a POMDP where the learning algorithm selects actions and receives observations and rewards in return. We start by defining a bundle from the predefined components. This time however, the policy of the user agent is defined as the random policy with action set :math:`\lbrace -5,-4,\dots{}, 4, 5`. Note the use of the override (``override_policy``) mechanism.

.. literalinclude:: ../../coopihc/examples/simple_examples/rl_sb3.py
   :language: python
   :linenos:
   :start-after: [start-define-bundle]
   :end-before: [end-define-bundle]

.. note::

    Finding an optimal policy in this case is actually straightforward: the optimal reward is obtained by minimizing the number of steps, which implies that if the goal is out of reach, an action of :math:`\pm 5` is selected, otherwise the remaining distance to the goal is selected. 

Making a Bundle compatible with Gym
--------------------------------------

The Gym API expects a few attributes and methods to be defined. We provide a wrapper which makes the translation between *CoopIHC* and gym called ``TrainGym``. We test that the environment is indeed Gym compatible with a function provided by stable baselines.

.. literalinclude:: ../../coopihc/examples/simple_examples/rl_sb3.py
   :language: python
   :linenos:
   :start-after: [start-define-traingym]
   :end-before: [end-define-traingym]

.. note::

    TrainGym converts discrete spaces to be in :math:`\\mathcal{N}` in line with Gym.

At this point, the environment is compatible with the Gym API, but we can not apply SB3 algorithms directly (``check_env`` with ``warn = False`` raises no warnings, but does with ``warn = True``). The reason is that *CoopIHC* returns dictionary spaces (``gym.spaces.Dict``) for actions , which is not supported by SB3 algorithms. We provide a simple `action wrapper <https://github.com/openai/gym/blob/master/gym/core.py#L318>`_ named ``TrainGym2SB3ActionWrapper`` that converts the actions to a "flattened space" (discrete actions are one-hot encoded to boxes).

.. literalinclude:: ../../coopihc/examples/simple_examples/rl_sb3.py
   :language: python
   :linenos:
   :start-after: [start-define-SB3wrapper]
   :end-before: [end-define-SB3wrapper]

It may be beneficial to write your own wrappers in many cases, especially considering it is usually pretty straightforward. The generic``TrainGym2SB3ActionWrapper`` wrapper converts discrete action spaces to unit boxes via a so-called one-hot encoding. The point of one-hot encoding is to make sure the metric information contained in the numeric representation does not influence learning (for example, for 3 actions A,B,C, if one were to code using e.g. A = 1, B = 2, C = 3, this could imply that A is closer to B than to C. Sometimes this needs to be avoided.) In the current example however, the actions represent distances to cover in either direction, and it is likely more efficient to convert the discrete space directly to a box by casting to a box (without one-hot encoding). In what follows, we will use hand-crafted wrappers.


Training an Agent
---------------------
There are various tricks to making DRL training more efficient, see for example `SB3's tips and tricks <https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html>`_. Usually, these require applying some extra wrappers. For example, for algorithms that work by finding the right parameters to Gaussians (e.g. PPO) it is recommended to normalize actions to :math:`[-1,1]`.

Below, we apply some wrappers that filters out the relevant information out of the observation space and casts it to a continuous space (if not, SB3 will one-hot encode it automatically). Then, we apply a wrapper that casts the action to a continuous space and normalizes it.

.. literalinclude:: ../../coopihc/examples/simple_examples/rl_sb3.py
   :language: python
   :linenos:
   :start-after: [start-define-mywrappers]
   :end-before: [end-define-mywrappers]


Not that everything is ready, we put all the relevant code into a function that, when called, will return an environment.

.. literalinclude:: ../../coopihc/examples/simple_examples/rl_sb3.py
   :language: python
   :linenos:
   :start-after: [start-make-env]
   :end-before: [end-make-env]

We are now ready to train the policy. Here, we use PPO with 4 vectorized environments:

.. literalinclude:: ../../coopihc/examples/simple_examples/rl_sb3.py
   :language: python
   :linenos:
   :start-after: [start-train]
   :end-before: [end-train]


A tensorboard excerpt shows that training is successful and rather quick, at 10 minutes of wall training time on a regular laptop.

.. figure:: images/rewards_rl.png
   :width: 800

   Average rewards per episode plotted against wall time. Less than 10 minutes are needed to train this simple policy.

Loading the Trained Policy in CoopIHC
----------------------------------------

To load the trained policy in CoopIHC, a special policy object called ``RLPolicy`` exists. It works by passing the agent's observation as input to the trained neural net and gathers the output action.

.. literalinclude:: ../../coopihc/examples/simple_examples/exploit_rlnet.py
   :language: python
   :linenos:
   :start-after: [start-load-policy]
   :end-before: [end-load-policy]

The policy can be visualized, which confirms training was successful:
    
.. literalinclude:: ../../coopihc/examples/simple_examples/exploit_rlnet.py
   :language: python
   :linenos:
   :start-after: [start-viz-policy]
   :end-before: [end-viz-policy]

.. figure:: images/trained_policy.png
   :width: 800

   Trained policy. As expected, the policy is the identity operator for the admissible actions, and otherwise it saturates at the edge admissible actions.

Finally, you can plug that policy back into the user and play with your bundle.

.. literalinclude:: ../../coopihc/examples/simple_examples/exploit_rlnet.py
   :language: python
   :linenos:
   :start-after: [start-play-policy]
   :end-before: [end-play-policy]