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

All agents are derived from the :ref:`BaseAgent <agent-label>` class, which provides a basic agent with an internal state, and the required API methods to be used by the modeler as well as by :ref:`bundles <bundles-label>`.

Upon instantiating a BaseAgent, the modeler should provide

* The role of the agent, (operator or assistant)
* The possible actions of the agent
* An observation engine
* An inference engine
