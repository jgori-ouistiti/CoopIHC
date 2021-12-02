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
All agents are derived from the :py:class:`BaseAgent <coopihc.agents.BaseAgent.BaseAgent>` class to ensure the compatibility with the :py:mod:`Bundle <coopihc.bundle>` Module.


When instantiating a :py:class:`BaseAgent <coopihc.agents.BaseAgent.BaseAgent>`, or any other agent for that matter, the modeler should provide

    * The role of the agent, ('user' or 'assistant')
    * The :py:class:`State <coopihc.spaces.State.State>` of the agent (defaults to an empty state)
    * The :py:mod:`Policy <coopihc.policy>` of the agent (defaults to a :py:class:`BasePolicy<coopihc.policy.BasePolicy.BasePolicy>`)
    * An :py:mod:`Observation Engine <coopihc.observation>` (defaults to a :py:class:`RuleObservationEngine<coopihc.policy.RuleObservationEngine.RuleObservationEngine>` if not provided)
    * An :py:mod:`Inference Engine <coopihc.inference>` (defaults to an :py:class:`BaseInferenceEngine<coopihc.inference.BaseInferenceEngine.BaseInferenceEngine>` if not provided)

Below is an example of instantiating a user with a py:class:`BaseAgent <coopihc.agents.BaseAgent.BaseAgent>`.

.. literalinclude:: ../../coopihc/examples/simple_examples/agents_examples.py
   :language: python
   :linenos:
   :start-after: [start-baseagent-init]
   :end-before: [end-baseagent-init]


Usually, it makes sense to define an entirely new agent, by subclassing the BaseAgent class, as in the :ref:`Quickstart example<quickstart-define-user-label>`.



Agents Zoo
----------------
