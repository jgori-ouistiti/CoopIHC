.. tasks:

Tasks
==================

.. start-quickstart-task

Tasks represent the passive component that the user is trying to drive to a given state. Usually in the *CoopIHC* context they will represent the part of an interface that the user can interact with.

Essentially, tasks are characterized by:

* An internal state called the **task state** which holds all the task's information e.g. the state of the interface.
* A **user step function**, which is a transition function that describes how the task state changes upon receiving a user action.
* An **assistant step function**, which is a transition function which describes how the task state changes based on the assistant action.

As an example, let's define a simple task where the goal of the user is to drive the substate called 'x' to a value of 4. Both the user and the assistant can provide three actions: -1, +0 and +1. We define a task by inheriting from ``InteractionTask`` and redefining a few methods.

.. literalinclude:: ../../coopihc/interactiontask/ExampleTask.py
    :pyobject: ExampleTask
    :linenos:


Some comments on the code snippet above:

+ The task state ``'x'`` is defined in the ``__init__`` method. Remember to always call ``super()``'s ``__init__`` before anything else.
+ The ``reset`` method resets the task to an initial state, in this case ``'x'=0``. You don't have to define a reset method, in which case it will inherit it from ``InteractionTask``, and the reset method will randomly pick values for each state.
+ You have to define a user and assistant step function otherwise an error will be raised. Both of these are expected to return the triple (task state, reward, is_done)
+ A render method is available if you want to render the task online. In this example, we simply print out the task state's value to the terminal, but you could plot some graphs or anything else.




.. TODO: link to check_task function once it exists 

.. You can verify that the task works as intended by bundling it with two ``BaseAgents`` (the simplest version of agents). Make sur that the actions spaces make sense, by specifying the policy for the two baseagents.

.. .. literalinclude:: ../../coopihc/examples/simple_examples/interactiontask_examples.py
..    :language: python
..    :linenos:
..    :start-after: [start-check-task]
..    :end-before: [end-check-task]

.. end-quickstart-task


ClassicControlTask
---------------------

PipeTaskWrapper
----------------

