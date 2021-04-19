.. agents:

Agents
==================

An agent is a combination of 4 components:

1. An observation engine, which produces observations for the agent, from the current game state.
2. An internal state, which is used to store parameters that belong to the agent and that are susceptible of being updated by the agent.
3. An inference engine, which uses observations to update the agent's internal state.
4. A policy, which describes what the possible actions of the agents are, and how the agent chooses them, based on its internal state and its current observation.


.. tikz:: Agent structure
    :include: tikz/agent.tikz
    :xscale: 100
    :align: center



BaseAgent
-----------------

All agents are derived from the :ref:`BaseAgent <agent-label>` class, which provides a basic agent with an internal state, and the required API methods to be used by the modeler as well as by :ref:`bundles <bundle-label>`.

Upon instantiating a BaseAgent, the modeler should provide

* The role of the agent, (operator or assistant)
* The possible actions of the agent
* An observation engine (defaults to a ``RuleObservationEngine`` if not provided)
* An inference engine (defaults to an ``InferenceEngine`` if not provided)

Below is an example of describing an operator with a BaseAgent.

.. code-block:: python

    ### Initialize a BaseAgent as operator
    action_set = [[-1,1]]
    action_space = [gym.spaces.Discrete(2)]
    observation_engine = RuleObservationEngine(BaseOperatorObservationRule)
    inference_engine = InferenceEngine(buffer_depth=0)
    my_operator = BaseAgent('operator', action_space, action_set, observation_engine = observation_engine, inference_engine = inference_engine)

Here is another example with a more complex action set:

.. code-block:: python

    ### Initialize a BaseAgent as operator with a more complex action space
    action_set = [[-1,1], None]
    action_space = [gym.spaces.Discrete(2), gym.spaces.Box(low = -1, high = 1, shape = (1,))]
    observation_engine = RuleObservationEngine(BaseOperatorObservationRule)
    inference_engine = InferenceEngine(buffer_depth=0)
    my_operator = BaseAgent('operator', action_space, action_set, observation_engine = observation_engine, inference_engine = inference_engine)


At this point, the operator is not ready for use, since it has not been bundled yet. Some methods and attributes can already be accessed

.. code-block:: python

    # Print out the action set of the operator
    >>> my_operator.action_set
    ([-1, 1], None)

    # Sample the policy (random) of the operator
    >>> my_operator.sample()
    (1, array([0.9568013], dtype=float32))

    # The agent is an operator
    >>> my_operator.role
    [0, 'operator']

    # The internal state is empty. Bundling later on will add an "Assistant Action" substate.
    >>> my_operator.state
    OrderedDict()

You can add internal states, and modify these states' values. For example, let us assume that our operator has a goal, and a 'Courage' parameter, which might influence its policy.

.. code-block:: python

    # Append a discrete and a continuous substate to the internal state
    # Specify possible values that the state can take in the discrete case
    my_operator.append_state('Goal', [gym.spaces.Discrete(2)], possible_values = [-1,1])
    # Don't specify possible values of the state for the continuous case
    my_operator.append_state('Courage', [gym.spaces.Box(low=-1, high=1, shape=(1, ))])

The possible values for the state are stored in a the ``state_dict`` attribute.

.. code-block:: python

    >>> my_operator.state_dict
    OrderedDict([('Goal', [-1, 1]), ('Courage', None)])


The state space is formed by combining each substate space, and is initialized with random admissible values

.. code-block:: python

    >>> my_operator.state_space
    Tuple(Discrete(2), Box(1,))

    >>> my_operator.state
    OrderedDict([('Goal', [1]), ('Courage', [array([-0.10157551], dtype=float32)])])


The value of each substate can be modified using the ``modify_state`` method. For example, let's set the 'Courage' parameter of this operator to its maximum value of 1, for a maximally courageous operator.

.. code-block:: python

    my_operator.modify_state('Courage', value = [1])

.. note::

    Values are assumed to be iterables (e.g. wrap floats and ints in a list)

Subclassing BaseAgent
------------------------

You can only do so much with the BaseAgent. Creating more elaborate operators (or assistants) is best achieved by subclassing the BaseAgent class. This ensures that your new agent has the required bundle API methods.

Usually, a custom agent will require redefining:


* the ``__init__()`` method, which describes how the instance of the class is created
* the ``finit()`` method, which optionally provides a way to finish initializing the instance after the task operator and assistants have been bundled
* the ``reset()`` method, which describes what should happen to that instance before starting a new run
* the ``sample()`` method which describes the policy of the agent
* the ``render()`` method which describes how to render the agent's information to the display.


In what follows, we explain the implementation of the ``GoalDrivenDiscreteOperator`` agent. This is an agent which has discrete actions driven by a goal it is trying to reach. It takes advantage of an operator model [link], which includes a policy, and is given upon initializing.

To define this operator, we are going to add an internal state called 'Goal', which can take any value in a possible set of 'Targets' (defined elsewhere, in the task).
It is further assumed that the agent will change goal after each run, but that it won't need to change the goal during a run. As such, no internal state change is needed during the run, which means there is no need to have an inference engine.


The ``__init__()`` method
"""""""""""""""""""""""""""
Use this method to initialize your class instance. It is recommended to call ``super().__init__()`` at some point.


In the ``GoalDrivenDiscreteOperator``, the action set and action spaces can be directly deduced from the operator model, and the inference engine is not required:

.. code-block:: python

    def __init__(self, operator_model, observation_engine = None):
        action_set = [operator_model.actions]
        action_space = [gym.spaces.Discrete(len(action_set))]
        self.operator_model = operator_model
        super().__init__(0, action_space, action_set, observation_engine = observation_engine, inference_engine = None)

.. note::

    If ``observation_engine = None`` is provided, the bundle will automatically assign a standard observation engine to the agent. If it is an operator, then it will be able to perfectly see the task state as well as its internal state, but none of the assistant state. The assistant follows the same rule with operator and assistant switched.

The ``finit()`` method
""""""""""""""""""""""""""""
Before the bundle has been initialized, the modeler first needs to:

* Initialize the task,
* Then initialize the operator,
* Then initialize the assistant.

The bundle then initializes. It first starts by assigning its reference to the task, operator and assistant, by providing them with a ``bundle`` attribute.
It then calls the operator and assistant's ``finit()`` method. This allows the modeler to initialize agents which depend on other agents. For example, the 'Goal' state can only take values in the possible 'Targets', which are defined in the task.

.. note::

    When used with a bundle, you can assume that the agent has a ``bundle`` attribute from which to access the task, operator, and assistant (``bundle.task, bundle.operator, bundle.assistant``) within any method except ``__init__()``

.. code-block:: python

    def finit(self):

        targets = self.bundle.task.state['Targets']
        self.append_state('Goal', [gym.spaces.Discrete(len(targets))], possible_values = targets)
        return


The ``reset()`` method
""""""""""""""""""""""""
The reset method should reset the internal state and any other attribute maintained by an agent during a run to pertinent initial values. Here, we will will a new target at random and assign it to the 'Goal' substate

.. code-block:: python

    def reset(self):

        targets = self.bundle.task.state['Targets']
        goal = numpy.random.choice(targets)
        self.modify_state('Goal', possible_values = targets, value = [goal])


The ``sample()`` method
"""""""""""""""""""""""""""
The sample method describes the policy of the agent. By default a random policy is provided to the agent. In this example, the policy is included in the operator model, and can be called by calling its sample method, see [link].

.. code-block:: python

    def sample(self):

        actions = self.operator_model.sample(self.observation_engine.observation)
        if isinstance(actions, (int, float)):
            actions = [actions]
        return actions


The ``render()`` method
""""""""""""""""""""""""""
This method describes how the agent's information is displayed to the display.
The signature of the render method should be ``def render(self, *args, mode="mode")``, where args is a tuple of the three axes (task, operator and assistant).
It is recommended to provide a 'plot' mode and a 'text' mode.

The 'plot' mode describes which information is plotted how on which axes, while the 'text' mode describes which information is directed to the terminal.

In the example below, we simply write out the goal state to the terminal and as a text label on the operator axes.

.. code-block:: python

    def render(self, *args, mode="mode"):

        if 'plot' in mode:
            axtask, axoperator, axassistant = args
            if self.ax is not None:
                pass
            else:
                self.ax = axoperator
                self.ax.text(0,0, "Goal: {}".format(self.state['Goal'][0]))
                self.ax.set_xlim([-0.5,0.5])
                self.ax.set_ylim([-0.5,0.5])
                self.ax.axis('off')
                self.ax.set_title(type(self).__name__ + " Goal")
        if 'text' in mode:
            print(type(self).__name__ + " Goal")
            print(self.state['Goal'][0])


Agents Zoo
------------------------

This list is ongoing and more agents will be added

Operators:

* The ``GoalDrivenDiscreteOperator`` [link], driven by a Goal and uses Discrete actions. It has to be used with a task that has a substate named Targets. Its internal state includes a goal substate, whose value is either one of the task's Targets. Uses an operator model[link] as policy.
* The ``GaussianContinuousBeliefOperator`` [link], maintains a continuous Gaussian belief. It can be used in cases where the goal of the operator is not directly observable to it.

Assistants:

* The ``DiscreteBayesianBeliefAssistant`` [link] An Assistant that maintains a discrete belief, updated with Bayes' rule. It supposes that the task has targets, and that the operator selects one of these as a goal.
