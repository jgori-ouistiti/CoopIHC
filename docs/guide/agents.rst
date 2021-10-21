.. agents:

Agents
==================

An agent is a combination of 4 components:

1. An observation engine, which produces observations for the agent, from the current game state.
2. An internal state, which is used to store parameters that belong to the agent and that are susceptible of being updated by the agent.
3. An inference engine, which uses observations to update the agent's internal state.
4. A policy, which describes what the possible actions of the agents are, and how the agent chooses them, based on its internal state and its current observation.


.. tikz:: Agent structure
    :include: tikz/agent.tikz
    :xscale: 100
    :align: center



BaseAgent
-----------------
All agents are derived from the :ref:`BaseAgent <base-agent-class-label>` class to ensure the compatibility with the :doc:`Bundle class<bundles>`.


When instantiating a BaseAgent, or any other agent for that matter, the modeler should provide

    * The role of the agent, ('user' or 'assistant')
    * The state of the agent (defaults to an empty state)
    * The policy of the agent (defaults to a :ref:`BasePolicy<base-policy-label>`)
    * An observation engine (defaults to a :ref:`RuleObservationEngine<rule-observation-engine-label>` if not provided)
    * An inference engine (defaults to an :ref:`BaseInferenceEngine<base-inference-engine-label>` if not provided)

Below is an example of instantiating a user with a BaseAgent.

.. literalinclude:: ../../core/agents.py
   :language: python
   :linenos:
   :start-after: [start-baseagent-init]
   :end-before: [end-baseagent-init]


Usually, it makes sense to define an entirely new agent, by subclassing the BaseAgent class, as in the :ref:`Quickstart example<quickstart-define-user-label>`.


.. _base-agent-class-label:

.. autoclass:: core.agents.BaseAgent
    :members:


Agents Zoo
------------------------

This list is ongoing

.. autoclass:: core.agents.GoalDrivenDiscreteUser

.. autoclass:: core.agents.LQRController

.. autoclass:: core.agents.FHDT_LQRController

.. autoclass:: core.agents.IHDT_LQRController

.. autoclass:: core.agents.IHCT_LQGController
