.. operator_model:

The Operator Model
========================

Operator policies are used in two cases:

* To describe the behavior of an operator, in which cases it is attached or passed to an operator
* To serve as a model of operator behavior, in which case it is attached or passed to an assistant.

In line with our design approach, we introduce a separate class to describe operator policies, which is called an operator model.


Operator models are essentially made up of two components:

* A likelihood function :math:`p(\text{action}| \text{observation, state})`, which describes the probability of an operator issuing an action given its last observation and the current game state.
* A ``sample()`` method, which describes how to select actions, based on the likelihood function.


List of implemented operator model classes (ongoing)
------------------------------------------------

* DiscreteOperatorModelMax: A class for discrete policies, which always selects the action which has highest likelihood. Redefine the ``compute_likelihood()`` method when subclassing this.


List of implemented operator models (ongoing)
------------------------------------------------
* GoalDrivenBinaryOperatorModel: A subclass of DiscreteOperatorModelMax, which expects the observation to have a 'Goal' state and a 'Position' state. A Binary Operator only has two actions: -action and action. It issues the intended action with probability .99 .
