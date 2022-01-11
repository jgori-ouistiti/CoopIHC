.. policy:

Policies
========================

Subclassing BasePolicy
-------------------------
You can define a new policy by subclassing the ``BasePolicy``; for which you can find an example in the :doc:`quickstart`. Other than that, there are a few predefined policies which you may find useful.


Explicit Likelihood Discrete (ELLD) Policy
--------------------------------------------
Explicit Likelihood Discrete (ELLD) Policy is used in cases where the agent model is straightforward enough to be specified by an analytical model.

Below, we define a simple probabilistic model which assigns different probabilities to each possible discrete action. Note that this function signature is what *CoopIHC* expects to find: in most cases the model will depend on at least the observation and on the particular action.

.. literalinclude:: ../../coopihc/examples/simple_examples/policy_examples.py
   :language: python
   :linenos:
   :start-after: [start-elld-def-model]
   :end-before: [end-elld-def-model]


You can then define your policy and attach the model to it:

.. literalinclude:: ../../coopihc/examples/simple_examples/policy_examples.py
   :language: python
   :linenos:
   :start-after: [start-elld-attach]
   :end-before: [end-elld-attach]


