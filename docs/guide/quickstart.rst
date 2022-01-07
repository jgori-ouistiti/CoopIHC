.. quickstart:

Quick Start
===============


Installation
---------------

*CoopIHC* is currently available on `TestPyPI <https://test.pypi.org/project/coopihc/>`_, although this might not be the latest version. You can install the package using pip with the following command:

.. code-block:: python

    python3 -m pip install --extra-index-url https://test.pypi.org/simple/ coopihc



You can also build directly from the github repository to get the latest version. To do so, install poetry, and run 

.. code-block:: shell

    poetry install

from within the folder. This will install *CoopIHC* in editable mode (basically equivalent to ``python3 -m pip install -e .``), together with all its dependencies.

Interaction Model
-------------------

*CoopIHC* builds on a :doc:`sequential two agent decision-making model<./interaction_model>`. You should read through the model to get a grasp of what each component does.


High-level view of CoopIHC code
--------------------------------

At a high level, your CoopIHC code will usually look like this

.. literalinclude:: ../../coopihc/examples/simple_examples/bundle_examples.py
   :language: python
   :linenos:
   :start-after: [start-highlevel-code]
   :end-before: [end-highlevel-code]


You will usually define a task, a user, an assistant, and bundle them together. You can then play several rounds of interaction until the game ends, and based on the collected data, you can do something.

States
------------
The interaction model uses the concept of states, for which *CoopIHC* introduces the ``Space`` ``StateElement`` and ``State`` objects. In the example below, a super-state is defined by a state, which itself is defined by two substates. Each of those substates holds a StateElement, which is a combination of a value and a space.

.. literalinclude:: ../../coopihc/examples/simple_examples/state_examples.py
   :language: python
   :linenos:
   :start-after: [start-state-example]
   :end-before: [end-state-example]


States, statelements subclass respectively the built-in dictionnary and the well-known NumPy arrays. This means that interacting with these objects should prove relatively familiar. To find out more about this and for extra details, go to :doc:`Space<space>`, :doc:`StateElement<stateelement>` and :doc:`State<state>`.


Tasks
--------
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

Agents
------------------
Agents are defined by four components:

* An internal state, which essentially gives memory to the agent
* An observation engine, which generates observation from the game state
* An inference engine, with which the agent modifies their internal state, essentially giving them the ability to learn
* A policy, used to take actions.

.. _quickstart-define-user-label:

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

Policies
------------
Defining a policy is done by subclassing the ``BasePolicy`` class and redefining the ``sample()`` method. Below, we show how ``ExamplePolicy`` used in the example before is defined.

.. literalinclude:: ../../coopihc/policy/ExamplePolicy.py
    :linenos:
    :pyobject: ExamplePolicy


.. note::

    The action that is returned has to be a valid ``StateElement``. Inside the policy, you can directly call ``self.new_action`` which returns a valid ``StateElement`` without a value, which you can just fill in, as done in the example.

.. note::

    Don't forget to return a reward with the action.

.. note::

    You can virtually put anything inside this function: that includes the output of a neural network, of a complex simulation process, and even the output of another bundle (see :doc:`modularity` for an example.)


Observation Engines
---------------------
Defining an observation engine is done by subclassing the ``BaseObservationEngine`` class and redefining the ``observe()`` method. Below, we show a basic example where an instance is defined that only looks at a particular substate (the code for the default observation engine is a little too verbose to be put here).

.. literalinclude:: ../../coopihc/observation/ExampleObservationEngine.py
    :linenos:
    :pyobject: ExampleObservationEngine


The effect of this engine can be tested by plugging in a simple State:

.. literalinclude:: ../../coopihc/examples/simple_examples/observation_examples.py
    :language: python
    :linenos:
    :start-after: [start-obseng-example]
    :end-before: [end-obseng-example]


.. note::

    Don't forget to return a reward with the observation.

.. note::

    You can virtually put anything inside this function: that includes the output of a neural network, of a complex simulation process, and even the output of another bundle (see :doc:`modularity` for an example.)


Inference Engines:
--------------------
Defining an Inference Engine is done by subclassing the ``BaseInferenceEngine`` class, and redefining the ``infer()`` method, as in previous components. Below, we define a new inference engine which has the exact same behavior as the ``BaseInferenceEngine``, for sake of illustration, and simply returns the agent's state without any modifications.


.. literalinclude:: ../../coopihc/inference/ExampleInferenceEngine.py
    :linenos:
    :pyobject: ExampleInferenceEngine






Bundles
---------------------
Bundles are the objects that join three components (task, user and assistant) to form the joint state of the game, collect the rewards and ensure a synchronous sequential sequences of observations, inferences and actions.

You have seen a couple of examples above where bundles are used, including their main methods: reset, step and render. 

In most cases, there is no need to define a new Bundle, and you can straightaway use the standard existing ``Bundle``.


.. note::

    Bundles also handle joint rendering as well as other practical things. More details can be found on :doc:`Bundle's reference page <./bundle>`


An overview of *CoopIHC*
-----------------------------------------------------
.. warning::

    Links below are outdated


1. Several implementations of user models, tasks and assistants exist in *CoopIHC* repository `*CoopIHC-Zoo* <https://github.com/jgori-ouistiti/CoopIHC-zoo>`_  
2. Several worked-out examples are given in this documentation. Those should give you a good idea about what can be done with *CoopIHC*.
 


