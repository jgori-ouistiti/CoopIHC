.. simulating_rollouts:

Assistants that simulate users
========================================

In many cases it is useful to have an assistant maintain a model of a user, for example to perform single-shot predictions (what will be the next user step?) or complete simulations by means of rollouts (what is the end result if I choose this policy throughout the task?).
This predictions can then be used in the decision-making process.


Since in *CoopIHC* we define users as classes (and/or instances), it seems natural to want to pass a user to an assistant, which could then query it to perform predictions. 




Single-shot predictions
-------------------------
Single-shot predictions are easy to implement with basic *CoopIHC* functions. Below is an example, where we consider a coordination task: the user and assistant have to select the same action to make the task state evolve (+1). The coordination will be successful because the assistant will manage a simulation of the user that provides single-shot predictions of the next user action.
 
We first modify the ``ExampleTask`` that we used in the Quickstart:

.. literalinclude:: ../../coopihc/interactiontask/ExampleTask.py
    :pyobject: CoordinatedTask
    :linenos:

The premise of the task is very simple: If the assistant and user input the same action (``self.user_action == self.assistant_action``), then the task state is incremented. When the task state reaches 9 the task is finished.

We then create a pseudorandom user, that uses a pseudorandom policy. It picks an action prescribed by the formula :math:`8 + p_0 \cdot{} x + p_1 \cdot{} x^2 + p_2 \cdot{} x^3 \pmod{10}` where the p's are user parameters and x is the task state.

.. literalinclude:: ../../coopihc/policy/ExamplePolicy.py
    :pyobject: PseudoRandomPolicy
    :linenos:


The assistant is then constructed as follows:

.. literalinclude:: ../../coopihc/agents/ExampleAssistant.py
    :pyobject: CoordinatedAssistant
    :linenos:

Notice that:

    * it expects that a ``user_model`` is passed during initialization.
    * it uses the ``finit`` mechanism to create a simulation that can be used by the assistant. That simulation is nothing more than a ``Bundle`` between the task and the user model.

This simulation is actually used in the policy of the assistant:


.. literalinclude:: ../../coopihc/policy/ExamplePolicy.py
    :pyobject: CoordinatedPolicy
    :linenos:

The policy of the assistant is straightforward:

    * Observe the current state of the game, and put the simulation in that state, via the reset mechanism.
    * Play the simulation just enough that the user model takes an action
    * Observe the action that was taken by the user model and pick the same.

At each turn, the assistant takes the same action as the user model. If we provide the assistant with the true model of the user, then the coordination is perfect:

.. literalinclude:: ../../coopihc/examples/simple_examples/assistant_has_user_model.py
    :linenos:
    :start-after: [start-user-model]
    :end-before: [end-user-model]




Rollouts
-----------
Usually, we need a more comprehensive simulation that spans several steps and that features a user and an assistant.
Using the same assistant simultaneously in a bundle and in a simulation is not straightforward, so *CoopIHC* provides a few helper classes. In particular, the inference engines, policies etc. used during simulation can not be the same as the ones during execution of the bundle (or, you would have infinite recursion). *CoopIHC* offers the possibility of having two different inference engines and policies, using the so-called ``DualInferenceEngine`` and ``DualPolicy``. Depending on the state of the engine, the primary or the dual engine is used (same for the policy).


To illustrate these, let's go over a variation of the previous example:

.. literalinclude:: ../../coopihc/examples/simple_examples/assistant_has_user_model.py
    :linenos:
    :start-after: [start-user-model-rollout]
    :end-before: [end-user-model-rollout]

Notice that the parameters of the ``PseudoRandomPolicy`` are given at initialization with the ``PseudoRandomUserWithParams`` (before, they were hard-coded in the user). If you look at the assistant, you see that we pass it a model of the task, a model of the user, as well as two parameters. These parameters are the last two parameters of the user model. The first one is unknown. The point of the assistant is now to infer that parameter using the models of the task and user it was given.

The code for the assistant is as follows:

.. literalinclude:: ../../coopihc/agents/ExampleAssistant.py
    :pyobject: CoordinatedAssistantWithRollout
    :linenos:

The state ``p0`` is the one that needs to be determined. Once it is known, the assistant can simply use the ``PseudoRandomPolicy`` to select the same action as the user.

The ``DualInferenceEngine`` holds two inference engines: the primary ``RolloutCoordinatedInferenceEngine`` which is used during the bundle execution, and the dual ``BaseInferenceEngine`` which is used for the simulation.

The remaining code is in the ``RolloutCoordinatedInferenceEngine``

.. literalinclude:: ../../coopihc/inference/RolloutCoordinatedInferenceEngine.py
    :pyobject: RolloutCoordinatedInferenceEngine
    :linenos:

First, we define a simulator object. For that, simply instantiate a ``Simulator`` as you would a ``Bundle``. The difference between a simulator and a bundle is that the former will consider the dual versions of the objects.
The inference is then straightforward: All possible values of ``p0`` are tested, and the correct one is the one that leads to the highest reward. 




