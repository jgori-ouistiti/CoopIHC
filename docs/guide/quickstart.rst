.. quickstart:

Quick Start
===============


Installation
---------------

You can install the package using pip with the following command:

.. code-block:: python

    pip install coopihc

Interaction Model
-------------------

*CoopIHC* builds on a sequential two agent decision-making model that is described :doc:`here<./interaction_model>`.


High-level view of CoopIHC code
--------------------------------

At a high level, CoopIHC code will usually look like this

.. literalinclude:: ../../coopihc/bundle/__init__.py
   :language: python
   :linenos:
   :start-after: [start-highlevel-code]
   :end-before: [end-highlevel-code]


It consists in defining tasks, users, assistants, bundling them together, and playing several rounds of interaction until the game ends.

States
------------
The interaction model heavily uses the concept of states, for which *CoopIHC* introduces the ``Space`` ``StateElement`` and ``State`` objects. In the example below, a super-state is defined by a state, which itself is defined by two substates.

.. literalinclude:: ../../coopihc/space.py
   :language: python
   :linenos:
   :start-after: [start-state-example]
   :end-before: [end-state-example]


To find out more about these, go to :doc:`Space, StateElement, State<space>`.


Tasks
--------
Tasks represent whatever the user is interacting with. They are essentially characterized by:

* An internal state, the **task state** which holds all the task's information.
* A **user step (transition) function**, which describes how the task state changes based on the user action.
* An **assistant step (transition) function**, which describes how the task state changes based on the assistant action.

As an example, let's define a simple task where the goal of the user is to drive the substate 'x' to a value of 4. Both the user and the assistant can provide three actions: -1, +0 and +1. As in most cases you will find, we define a task by inheriting from ``InteractionTask`` and redefining a few methods.

.. literalinclude:: ../../coopihc/interactiontask/__init__.py
    :pyobject: ExampleTask
    :linenos:



In ``__init__``, the task states are defined ('x' in this example); the ``reset`` method fixes the initial condition ('x'=0). The user and assistant step functions define how the task state evolves with respect to incoming user or assistant actions.
Finally, the render method is a way to display information to the screen. Here we considered a very simplistic output which prints the minimum information, but you could handles plots, animations etc.

.. note::

    The ``__init__`` method of a task should always be started with a call to ``super()``'s ``__init__``. This will be true for the other objects as well.

.. note::

    Notice how ``StateElement`` arithmetic simplifies the coding, making ``self.state['x'] += self.user_action`` possible.

.. note::

    Some methods have a default behavior inherited from the parent class. For example, if the ``reset()`` method is undefined in the child class, then the parent method is used, which uniformly samples a value from the states.



You can verify that the task works as intended by bundling it with two ``BaseAgents`` (the simplest version of agents). Make sur that the actions spaces make sense, by specifying the policy for the two baseagents.

.. literalinclude:: ../../coopihc/bundle/__init__.py
   :language: python
   :linenos:
   :start-after: [start-check-task]
   :end-before: [end-check-task]

Agents
------------------
Agents are defined by four components:

* Their internal state
* Their observation engine
* Their inference engine
* Their policy

.. _quickstart-define-user-label:

Defining a new agent is done by subclassing the ``BaseAgent`` class:

.. literalinclude:: ../../coopihc/agents/__init__.py
    :linenos:
    :pyobject: ExampleUser


.. note::

    All 4 components default to their corresponding base implementation if not provided.

You can check that the user model works as intended by bundling it with the task. Below, we try it out without an assistant, so we modify the task very simply by redefining its ``assistant_step()`` method.

.. literalinclude:: ../../coopihc/bundle/__init__.py
    :language: python
    :linenos:
    :start-after: [start-check-taskuser]
    :end-before: [end-check-taskuser]

Policies
------------
Defining a policy is done by subclassing the ``BasePolicy`` class and redefining the ``sample()`` method. Below, we show how ``ExamplePolicy`` used in the example before is defined.

.. literalinclude:: ../../coopihc/policy.py
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

.. literalinclude:: ../../coopihc/observation/__init__.py
    :linenos:
    :pyobject: ExampleObservationEngine


The effect of this engine can be tested by plugging in a simple State:

.. code-block::

    Game state before observation
      Index  Label                    Value    Space
    -------  -----------------------  -------  ----------------------------------
          0  substate1|substate_x|0   None     Space[Continuous((1, 1)), float64]
          1  substate1|substate_y|0   [[2]]    Space[Discrete([3]), int64]
          2  substate_2|substate_a|0  None     Space[Discrete([3]), int8]
    Observation
      Index  Label         Value    Space
    -------  ------------  -------  ----------------------------------
          0  substate_x|0  None     Space[Continuous((1, 1)), float64]
          1  substate_y|0  [[2]]    Space[Discrete([3]), int64]


.. note::

    Don't forget to return a reward with the observation.

.. note::

    You can virtually put anything inside this function: that includes the output of a neural network, of a complex simulation process, and even the output of another bundle (see :doc:`modularity` for an example.)


Inference Engines:
--------------------
Defining an Inference Engine is done by subclassing the ``BaseInferenceEngine`` class, and redefining the ``infer()`` method, as in previous components. Below, we define a new inference engine which has the exact same behavior as the ``BaseInferenceEngine``, for sake of illustration, and simply returns the agent's state without any modifications.


.. literalinclude:: ../../coopihc/inference/__init__.py
    :linenos:
    :pyobject: ExampleInferenceEngine






Bundles
---------------------
Bundles are the objects that join the states of the three components (task, user and assistant) to form the joint state of the game, collect the rewards and ensure a synchronous sequential sequences of observations, inferences and actions.

You have seen a couple of examples above where bundles are used, including their main methods: reset, step and render.


.. note::

    Bundles also handle joint rendering as well as other practical things.




.. warning::

    Documentation is out of date below

An overview of *CoopIHC*
-----------------------------------------------------

1. *CoopIHC* comes equipped with presently two tasks (pointing with a cursor, and a human eye-gaze selection task). Look at the list of modules [link].
2. Several operators and assistants are provided, some generic and described in the agent sections [link], others adapted to one of the tasks, described in the modules [link]
3. Several bundles are provided, that cover many use cases. These are described in the bundles section [link]
4. One can define new agents by minimally writing new code, by taking advantage of the modular approach of *CoopIHC*. In particular, inference engines [link], observation engines [link], and operator models [link] can be re-used and sub-classed.


What's next?
------------------------
 1. Build you own tasks [link]
 2. Build you own agents [link]
 3. Train, evaluate, simulate [link]
