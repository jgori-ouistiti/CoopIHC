.. agents:

Agents
==================

.. start-quickstart-agent

Agents are defined by four components:

* An internal state, which essentially gives memory to the agent;
* An observation engine, which generates observations from the game state, giving the agent the ability to perceive;
* An inference engine, with which the agent modifies its internal state, giving it the ability to learn;
* A policy, used to take actions, giving the agent the ability to make decisions.


.. tikz:: Agent structure
    :include: tikz/agent.tikz
    :xscale: 100
    :align: center



You define a new agent by subclassing the ``BaseAgent`` class. As an example, we now create an agent which goes with the ExampleTask that we defined in :doc:`Tasks<tasks.rst>`. We make an agent with a ``'goal'`` state to indicate the value for ``'x'`` that it wants to achieve, and make its available actions :math:`[-1,0,1]`. These actions are chosen via the ``ExamplePolicy`` (see :doc:`Policies<policy.rst>`).

.. literalinclude:: ../../coopihc/agents/ExampleUser.py
    :linenos:
    :pyobject: ExampleUser


.. note::

    All 4 components default to their corresponding base implementation if not provided.

You can verify that the user model works as intended, by bundling it with the task. Since we haven't provided an assistant yet, we slightly change the task very, redefining its ``on_assistant_action()``.

.. literalinclude:: ../../coopihc/examples/basic_examples/bundle_examples.py
    :language: python
    :linenos:
    :start-after: [start-check-taskuser]
    :end-before: [end-check-taskuser]

.. end-quickstart-agent


LQRControllers
--------------------
Not documented yet, see API Reference