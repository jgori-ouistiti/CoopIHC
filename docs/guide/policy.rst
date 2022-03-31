.. policy:

Policies
========================

Subclassing BasePolicy
-------------------------

.. start-quickstart-policy

To define a policy, simply subclass `:py:class:BasePolicy<coopihc.policy.BasePolicy>` and redefe its ``sample()`` method. Below, we show how ``ExamplePolicy`` has been defined.

.. literalinclude:: ../../coopihc/policy/ExamplePolicy.py
    :linenos:
    :pyobject: ExamplePolicy


.. note::

    Don't forget to return a reward with the action.

.. note::

    You can virtually put anything inside this function: that includes the output of a neural network, of a complex simulation process, and even the output of another bundle (see :doc:`modularity` for an example.)

.. end-quickstart-policy

Other than that, there are a few predefined policies which you may find useful.


Explicit Likelihood Discrete (ELLD) Policy
--------------------------------------------
Explicit Likelihood Discrete (ELLD) Policy is used in cases where the agent model is straightforward enough to be specified by an analytical model.

Below, we define a simple probabilistic model which assigns different probabilities to each possible discrete action. Note that this function signature is what *CoopIHC* expects to find: in most cases the model will depend on at least the observation and on the particular action.

.. literalinclude:: ../../coopihc/examples/basic_examples/policy_examples.py
   :language: python
   :linenos:
   :start-after: [start-elld-def-model]
   :end-before: [end-elld-def-model]


You can then define your policy and attach the model to it:

.. literalinclude:: ../../coopihc/examples/basic_examples/policy_examples.py
   :language: python
   :linenos:
   :start-after: [start-elld-attach]
   :end-before: [end-elld-attach]


BIGDiscretePolicy
--------------------
The Bayesian Information Gain Policy is a reimplementation of a technique introduced by Liu et al [1]_.

The main ideas/assumptions are:

   * A user wants the task to go to some goal state :math:`\\Theta`
   * The assistant can put the task in a number of states (X)
   * The user can perform a given set of action Y
   * A model :math:`p(Y=y|X=X, \\Theta = \\theta)` exists for user behavior that is leveraged by the assistant. Note that this model is not necessarily correct.

After the policy has been defined, make sure to call:

   * ``attach_set_theta``, to specify the potential goal states
   * ``attach_transition_function``, to specify how the task state evolves after an assistant action.


You can find an example implementation in CoopIHC-Zoo's pointing module. Below are the important steps:

.. code-block:: python

   TASK_SIZE = 30
   TARGETS = [1,5,6,19]

   action_state = State()
   action_state["action"] = StateElement(
      0,
      autospace([i for i in range(TASK_SIZE)]),
      out_of_bounds_mode="error",
   )
   # Define the user_policy_model that the assistant will use
   user_policy_model = XXX

   # Define Policy
   agent_policy = BIGDiscretePolicy(action_state, user_policy_model)

   # Specify the potential Goal states of the user. Here, potential goals are all cases where targets may be the use goal
   set_theta = [
            {
                ("user_state", "goal"): StateElement(
                    t,
                    discrete_space(numpy.array(list(range(TASK_SIZE)))),
                )
            }
            for t in TARGETS
        ]
   # Attach this set to the policy
   self.policy.attach_set_theta(set_theta)

   # Define the predicted future observation of the user due to assistant action
   def transition_function(assistant_action, observation):
      """What future observation will the user see due to assistant action"""
      # always do this
      observation["assistant_action"]["action"] = assistant_action
      # specific to BIGpointer
      observation["task_state"]["position"] = assistant_action
      return observation

   # Attach it to the policy
   self.policy.attach_transition_function(transition_function)



LinearFeedback
-----------------

RLPolicy
---------

The RLPolicy is a wrapper for a neural network trained via Deep Reinforcement Learning. For an example, head over to :ref:`Using Reinforcement Learning`.


WrapAsPolicy
-------------




















.. [1] Liu, Wanyu, et al. "Bignav: Bayesian information gain for guiding multiscale navigation." Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems. 2017.