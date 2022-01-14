.. agents:

Agents
==================

.. start-quickstart-agent

Agents are defined by four components:

* An internal state, which essentially gives memory to the agent
* An observation engine, which generates observation from the game state
* An inference engine, with which the agent modifies their internal state, essentially giving them the ability to learn
* A policy, used to take actions.


.. tikz:: Agent structure
    :include: tikz/agent.tikz
    :xscale: 100
    :align: center



You define a new agent is by subclassing the ``BaseAgent`` class. For example, we can create an agent which goes with the task that we just defined. The agent has a ``'goal'`` state to indicate how much it wants ``'x'`` to be, and its available actions are [-1,+0,+1]. How these actions are chosen depends on the policy of this agent, which is here instantiated with an ``ExamplePolicy`` (more on this below).

.. literalinclude:: ../../coopihc/agents/ExampleUser.py
    :linenos:
    :pyobject: ExampleUser


.. note::

    All 4 components default to their corresponding base implementation if not provided.

You can check that the user model works as intended by bundling it with the task. Below, we try it out without an assistant, so we modify the task very simply by redefining its ``assistant_step()`` method.

.. literalinclude:: ../../coopihc/examples/simple_examples/bundle_examples.py
    :language: python
    :linenos:
    :start-after: [start-check-taskuser]
    :end-before: [end-check-taskuser]

.. end-quickstart-agent


LQRControllers
--------------------
Not documented yet, see API Reference