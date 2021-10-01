.. user_model:

The User Model
========================

User policies are used in two cases:

* To describe the behavior of an user, in which cases it is attached or passed to an user
* To serve as a model of user behavior, in which case it is attached or passed to an assistant.

In line with our design approach, we introduce a separate class to describe user policies, which is called an user model.


User models are essentially made up of two components:

* A likelihood function :math:`p(\text{action}| \text{observation, state})`, which describes the probability of an user issuing an action given its last observation and the current game state.
* A ``sample()`` method, which describes how to select actions, based on the likelihood function.


List of implemented user model classes (ongoing)
------------------------------------------------

* DiscreteUserModelMax: A class for discrete policies, which always selects the action which has highest likelihood. Redefine the ``compute_likelihood()`` method when subclassing this.


List of implemented user models (ongoing)
------------------------------------------------
* GoalDrivenBinaryUserModel: A subclass of DiscreteUserModelMax, which expects the observation to have a 'Goal' state and a 'Position' state. A Binary User only has two actions: -action and action. It issues the intended action with probability .99 .
