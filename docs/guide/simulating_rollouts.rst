.. simulating_rollouts:

Assistants that simulate users
========================================

In many cases it is useful to have an assistant maintain a model of a user, for example to perform single-shot predictions (what will be the next user step?) or complete simulations by means of rollouts (what is the end result if I choose this policy throughout the task?).
This predictions can then be used in the decision-making process.

Since in *CoopIHC* we define users as classes (and/or instances), it seems natural to want to pass a user to an assistant, which could then query it to perform predictions. The examples below show a few ways in which this can be done.



Single-shot predictions
-------------------------

For this example, we consider a simple case where the user and assistant have to coordinate their actions to make the task state evolve. The coordination will be successful because the assistant will manage a simulation of the user that provides single-shot predictions of the next user action.
 
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


Single-shot predictions and inference
----------------------------------------

Usually, it is impossible to specify the true user model. In this coordination task, a mismatch between the true user and the user model crashes the performance of the assistant. To see this, run the following code, where ``PseudoRandomUserWithParams(p=[1, 5, 7])`` is the pseudorandom user where the parameters of its policy are specified during initialization (here :math:`p_0 = 1`, :math:`p_1 = 5`, :math:`p_2 = 7`).

.. literalinclude:: ../../coopihc/examples/simple_examples/assistant_has_user_model.py
    :linenos:
    :start-after: [start-user-model-mismatch]
    :end-before: [end-user-model-mismatch]

The agents are uncoordinated and the task ends because of the hard limit of 100 rounds that we have specified. 


Mismatches in models can be devastating. Yet, they can be resolved in some cases with certain assumptions.
For example, let's assume that the assistant knows the formula of the user's policy but not the parameter :math:`p_0`.  
Here is what we are going to do:

    1. modify the assistant to have a ``user_p0`` state
    2. update ``user_p0`` by an appropriate inference engine based on the observation of the user's behavior
    3. pass the updated ``user_p0`` to the user model 
    4. use the updated model to perform predictions.

Steps 2--4 may have to be repeated in more complex cases, but the example we tackle here is quite straightforward and we will be able to infer :math:`p_0` perfectly on our first attempt.



First, we define the new Assistant, simply adding a new state ``user_p0`` and a new inference engine ``CoordinatedInferenceEngine``.

.. literalinclude:: ../../coopihc/agents/ExampleAssistant.py
    :pyobject: CoordinatedAssistantWithInference
    :linenos:

The inference engine works simply by comparing the prediction of the user model with the actual observation of the user action. The parameter is tuned until the prediction matches the observation

.. literalinclude:: ../../coopihc/inference/ExampleInferenceEngine.py
    :pyobject: CoordinatedInferenceEngine
    :linenos:

You can check that the performance is as good as the first case where the true model was given by running the code below, since inference is guaranteed to be successful on the first try.


.. literalinclude:: ../../coopihc/examples/simple_examples/assistant_has_user_model.py
    :linenos:
    :start-after: [start-user-model-inference]
    :end-before: [end-user-model-inference]

Rollouts
-----------


Example not finished. There are several ways of doing this.  One of them is something like:

.. code-block:: python

    # Define a simulation bundle that is passed to a CoopIHC component e.g. an inference engine
    simulation_bundle = Bundle(
        task=task_model, user=user_model, assistant=assistant_simulation
    )

    # Inside the inference engine, plug in the observed states inside the bundle and run it. Then do something based on that information.
    reset_dic = copy.deepcopy(self.observation)
    del reset_dic["assistant_state"]

    reset_dic = {
        **reset_dic,
        **{
            "user_state": {
                "p0": numpy.array([[i]]),
                "p1": self.state.user_p1[:],
                "p2": self.state.user_p2[:],
            }
        },
    }

    self.simulation_bundle.reset(turn=0, dic=reset_dic)
    while True:
        state, rewards, is_done = self.simulation_bundle.step()
        rew[i] += sum(rewards.values())
        if is_done:
            break





