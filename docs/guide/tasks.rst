.. tasks:

Tasks
==================

.. start-quickstart-task

Tasks represent the agent's environment. Usually in the *CoopIHC* context, the task will represent the part of an interface that the user can interact with and drive to a certain state.

Essentially, tasks are characterized by:

* An internal state called the **task** ``state`` which holds all the task's information; for example, the state of the interface.
* A ``on_user_action()`` method, which is a transition function that describes how the task state changes on receiving a user action.
* An ``on_assistant_action()`` method, which is a transition function that describes how the task state changes based on the assistant action.

As an example, let's define a simple task where the goal of the user is to drive the substate called 'x' to a value of 4. Both the user and the assistant can provide three actions: -1, +0 and +1. We define a task by inheriting from ``InteractionTask`` and redefining a few methods.

.. literalinclude:: ../../coopihc/interactiontask/ExampleTask.py
    :pyobject: ExampleTask
    :linenos:


Some comments on the code snippet above:

    + The task state ``'x'`` is defined in the ``__init__`` method. Remember to always call ``super()``'s ``__init__`` before anything else to ensure all necessary variables internal to *CoopIHC* are set.
    + The ``reset`` method resets the task to an initial state, in this case ``'x'=0``. You don't have to define a reset method, in which case it will inherit it from `:py:class:InteractionTask<coopihc.interactiontask.InteractionTask>`, and the reset method will randomly pick values for each state.
    + You have to define a user and assistant step function otherwise an error will be raised. Both of these are expected to return the triple (task state, reward, is_done).
    + A render method is available if you want to render the task online, see `:py:class:InteractionTask<coopihc.interactiontask.InteractionTask>`


.. TODO: link to check_task function once it exists 

.. You can verify that the task works as intended by bundling it with two ``BaseAgents`` (the simplest version of agents). Make sur that the actions spaces make sense, by specifying the policy for the two baseagents.

.. .. literalinclude:: ../../coopihc/examples/basic_examples/interactiontask_examples.py
..    :language: python
..    :linenos:
..    :start-after: [start-check-task]
..    :end-before: [end-check-task]

.. end-quickstart-task


More control mechanisms
-------------------------
There are a few more control mechanisms:

    * ``finit``, which stands for finish initialization, and acts as an extra ``__init__`` function that is called by the bundle. It is useful when the initialization of the task or agents depend on one another. For example 

    .. code-block:: python

        def finit(self):
            self.a = self.bundle.user.state['goal'].shape

    * ``on_bundle_constraints`` which is called by a bundle after the finits. Its purpose is to enforce task constraints that valid agents should have. It should return errors or nothing, for example:

    .. code-block:: python

        def on_bundle_constraints(self):
            if not hasattr(self.bundle.user.state, "goal"):
                raise AttributeError(
                    "You must pair this task with a user that has a 'goal' state"
                )


ClassicControlTask
---------------------

PipeTaskWrapper
----------------

