.. api:

Interaction-Agents API
========================

To function with bundles and other components, each new component should be compatible with bundles, which implies that each new object follows certain rules.

States
------------

States hold the information about the state value and spaces. They are used by policies to store actions, by agents to store internal states, and by bundles to generate the aggregated states.

You should not have to modify this class. In *interaction-agents*, states are derived from the ``OrderedDict`` class, a dictionary which guarantees the same order when iterating over it.

A state can be composed of several entries, each entry (key, values) is a substate named 'key' and which takes values 'values'.

Values are:

* either States
* or a list of length 3  [value, space, possible_values]. All three items of the list may be entered as None at some point.


States are also assumed to implement the following methods:

* Any method from OrderedDict
* a modified __getitem__() method which allows one to use two extra forms of indexation:
    * Indexation by int (e.g. State[0] will return the first substate of State)
    * Indexation with special tags (e.g. State['_value_x'] will return State['x'][0])
* a modified __setitem__() method which allows one to use extra forms of assigments depending on the type of value:
    * if value is a State, then call the regular __setitem__
    * if the key uses special tags (e.g. State['_value_x'] = [1]), then it is assumed that you want to change the value of the substate 'x' (which is equivalent to State['_value_x'][0] = [1]).
    * if the value is a list of length 3, and one or more items are None, each of them are replaced by the current value of State. For example, if before assignment we have State['x'] = [[0], [gym.spaces.Discrete(2)], [-1,1]]. The assignment State['x'] = [[1], None, None] will lead to State['x'] = [[1], [gym.spaces.Discrete(2)], [-1,1]]
* A reset() method, which changes the value of the State by sampling (using the sample() method on) each space.

.. warning::

    In order not to break the reference between internal states and the encapsulating states, you should only ever change the contents of the states instead of assigning new substates altogether.


Policies
-------------

Required attributes for a policy are:

* ``action_state``, which is an instance of ``State()``, and stores information about an agent's action.

Required methods:

* to have a bound method called ``sample()``, which describes how the agent will choose an action.

Additional information:

* Policies are assigned a host when attached to an operator or assistant. That way, the policy can look at the agent's internal state (host.state), its observation (host.inference_engine.buffer[-1]).
* Policies should return an iterable (usually a list, or a numpy array)
* You can create a new policy by subclassing ``Policy``.

Observation Engines
----------------------

Required attributes:

* a type attribute (either ``base``, ``rule``, or ``process``)

Required methods:

* an observe method, which takes as input a game state (``State`` object), and outputs an observation (another ``State`` object) and the associated reward (a float)

You can create a new observation engine by subclassing ``ObservationEngine``.

Inference Engines
-------------------

Required attributes:

* to have a ``buffer`` attribute, whose items can be accessed via indexing (list-like)

Required methods:

* to have a bound method called add_observation, which takes as input an observation which it adds to its buffer
* to have an ``infer`` method which returns a new value for the internal state along with the associated reward.

You can create a new observation engine by subclassing ``InferenceEngine``.

.. warning::

    This class is outdated and has not been tested

Agents
---------------

Agents should only ever be created by subclassing the ``BaseAgent`` class.

Different behaviors can be obtained from an agent by redefining the following bound methods:

* ``finit()`` which is run after the three components (task + 2 agents) of bundles have been initialized, which allows one to have initialization of agents dependent on the other components. When bundled, the agent has a bundle attribute, from which it can query properties of other components (e.g. an operator can call ``self.bundle.task.size_of_some_property_`` if it requires that size to initialize, say, its internal state)
* ``reset()``, which by defaults resets the agent's internal state by calling its states' reset.
* ``render()``, which describes what the agent should be render to the display

BaseAgent are initialized by calling their __init__() method, which expects:

* A role string ('operator' or 'assistant')
* A State() object (if not provided, empty State)
* A Policy() object (if not provided, will likely fail -- should provide a default ?)
* An ObservationEngine() object (if not provided, will fall back to an engine which sees all substates except those of the other agent)
* An InferenceEngine() object (if not provided, will not update state)


Tasks
---------------------

Tasks should be created by subclassing the ``InteractionTask``.

Different behavior can be obtained from a task by redefining the following bound methods:

* ``finit()``, ``reset()``, ``render()`` with similar description as for the agent
* ``operator_step()`` and ``assistant_step()``, which returns the new value of the task state, as well as the associated rewards, a boolean value to indicate whether or not the task has finished, after respectively an operator action or an assistant action.


Bundles
---------------------
New bundles can be created by subclassing the ``Bundle`` object, and redefining the following methods:

* step()
* reset()
* close()
