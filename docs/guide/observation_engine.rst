.. observation_engine:

The Observation Engines
========================
.. start-quickstart-obseng-intro

In an interactive setting, states are rarely perfectly observable by the various agents:

    * the other agent's internal states are unknown,
    * the task's state may be partially observable; e.g. a human observer is imperfect and produces noisy observations,
    * an agent's own internal state may be partially observable; e.g. a human observer with decaying memory

Furthermore, there might be a cost associated with making observations:

    * For example, there can be a tradeoff between the time needed to produce an observation and its quality: Precise observations may be costly (in terms of time).
    * A human observer may enjoy making observations which are very different (according to some criterion) to the previous ones, in which case it would be rewarded for differing observations, satisfying its curiosity.

*CoopIHC* provides a generic object called an observation engine which specifies how an observation is created from the game state. To create a new observation engine, you can use an existing observation engine or subclass the ``BaseObservationEngine``.

.. end-quickstart-obseng-intro


Subclassing ``BaseObservationEngine``
--------------------------------------
.. start-quickstart-obseng-subclass

To create a new engine by subclassing the ``BaseObservationEngine`` class, you simply have to redefine the ``observe()`` method. You can virtually put anything inside this function: that includes the output of a neural network, of a complex simulation process, and even the output of another bundle (see :doc:`modularity` for an example). Below, we show a basic example we define an engine that only looks at a particular substate.

.. literalinclude:: ../../coopihc/observation/ExampleObservationEngine.py
    :linenos:
    :pyobject: ExampleObservationEngine

Don't forget to return a reward with the observation. The effect of this engine can be tested by plugging in a simple State:

.. literalinclude:: ../../coopihc/examples/simple_examples/observation_examples.py
    :language: python
    :linenos:
    :start-after: [start-obseng-example]
    :end-before: [end-obseng-example]


.. note::

    The signature ``observe(self, game_state=None)`` is expected. When called with ``game_state = None``, the engine will fetch the agent's observation. If the game state is actually passed, it will user the input state as basis to produce the observation. This is useful e.g. when testing your engine and you want to control the input.

.. end-quickstart-obseng-subclass


Combining Engines -- CascadedObservationEngine
-----------------------------------------------

Serially combine several engines. Not documented yet, see API Reference


WrapAsObservationEngine
------------------------

Wrap a bundle as an engine. Not documented yet, see API Reference

RuleObservationEngine
------------------------
This observation engine is specified by rules regarding each particular substate, using a so called mapping.

.. code-block:: python

    obs_eng = RuleObservationEngine(mapping=mapping)
    obs, reward = obs_eng.observe(game_state=example_game_state())

For example, in the example below, the observation engine is defined in a way that it will not observe the first substate, that it will have a noisy observation of the second substate, and that it will perfectly observe the remainder of the state.

.. tikz:: The observation engine
    :include: tikz/observation_engine.tikz
    :xscale: 100
    :align: center

A mapping is any iterable where an item is:

.. code-block:: python

    (substate, subsubstate, _slice, _func, _args, _nfunc, _nargs)

The elements in this mapping are applied to create a particular component of the observation space, as follows

.. code-block:: python

    observation_component = _nfunc(_func(state[substate][subsubstate][_slice], _args), _nargs)

which are then collected to form an observed state. For example, a valid mapping for the ``example_game_state`` mapping that states that everything should be observed except the game information is as follows:

.. code-block:: python

    from coopihc.space.utils import example_game_state
    print(example_game_state())

    # Define mapping
    mapping = [
        ("game_info", "turn_index", slice(0, 1, 1), None, None, None, None),
        ("game_info", "round_index", slice(0, 1, 1), None, None, None, None),
        ("task_state", "position", slice(0, 1, 1), None, None, None, None),
        ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), None, None, None, None),
        ("assistant_state", "beliefs", slice(0, 8, 1), None, None, None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]

    # Apply mapping
    obseng = RuleObservationEngine(mapping=mapping)
    obseng.observe(example_game_state())

As a more complex example, suppose we want to have an observation engine that behaves as above, but which doubles the observation on the ("user_state", "goal") StateElement. We also want to have a noisy observation of the ("task_state", "position") StateElement. We would need the following mapping:

.. code-block:: python

    def f(observation, gamestate, *args):
        gain = args[0]
        return gain * observation

    def g(observation, gamestate, *args):
        return random.randint(0, 1) + observation

    mapping = [
        ("task_state", "position", slice(0, 1, 1), None, None, g, ()),
        ("task_state", "targets", slice(0, 2, 1), None, None, None, None),
        ("user_state", "goal", slice(0, 1, 1), f, (2,), None, None),
        ("user_action", "action", slice(0, 1, 1), None, None, None, None),
        ("assistant_action", "action", slice(0, 1, 1), None, None, None, None),
    ]

.. note::

    It is important to respect the signature of the functions you pass in the mapping (viz. f and g's signatures).


Typing out a mapping may be a bit laborious and hard to comprehend for collaborators; there are some shortcuts that make defining this engine easier.

Example usage:

.. code-block:: python

    obs_eng = RuleObservationEngine(
        deterministic_specification=engine_specification,
        extradeterministicrules=extradeterministicrules,
        extraprobabilisticrules=extraprobabilisticrules,
    )

There are three types of rules:

1. Deterministic rules, which specify at a high level which states are observable or not, e.g.

.. code-block :: python

    engine_specification = [
            ("game_info", "all"),
            ("task_state", "targets", slice(0, 1, 1)),
            ("user_state", "all"),
            ("assistant_state", None),
            ("user_action", "all"),
            ("assistant_action", "all"),
        ]

2. Extra deterministic rules, which add some specific rules to specific substates

.. code-block:: python

    def f(observation, gamestate, *args):
        gain = args[0]
        return gain * observation

    f_rule = {("user_state", "goal"): (f, (2,))}
    extradeterministicrules = {}
    extradeterministicrules.update(f_rule)

3. Extra probabilistic rules, which are used to e.g. add noise

.. code-block :: python

    def g(observation, gamestate, *args):
        return random.random() + observation

    g_rule = {("task_state", "position"): (g, ())}
    extraprobabilisticrules = {}
    extraprobabilisticrules.update(g_rule)





.. warning ::

    This observation engine handles deep copies, to make sure operations based on observations don't mess up the actual states. This might be slow though. If you want to get around this, you could subclass the RuleObservationEngine to remove copies.





Several rules are predefined:

+----------------+------------+-------------+-------------+------------------+--------------+-------------------+--------------------------------------+
| Rule Name      | Game Info  | Task State  | User State  | Assistant State  | User Action  | Assistant Action  | Full name                            |
+================+============+=============+=============+==================+==============+===================+======================================+
| Oracle         | |tick|     | |tick|      | |tick|      | |tick|           | |tick|       | |tick|            | oracle_engine_specification          |
+----------------+------------+-------------+-------------+------------------+--------------+-------------------+--------------------------------------+
| Blind          | |tick|     | |cross|     | |cross|     | |cross|          | |tick|       | |tick|            | blind_engine_specification           |
+----------------+------------+-------------+-------------+------------------+--------------+-------------------+--------------------------------------+
| BaseTask       | |tick|     | |tick|      | |cross|     | |cross|          | |tick|       | |tick|            | base_task_engine_specification       |
+----------------+------------+-------------+-------------+------------------+--------------+-------------------+--------------------------------------+
| BaseUser       | |tick|     | |tick|      | |tick|      | |cross|          | |tick|       | |tick|            | base_user_engine_specification       |
+----------------+------------+-------------+-------------+------------------+--------------+-------------------+--------------------------------------+
| BaseAssistant  | |tick|     | |tick|      | |cross|     | |tick|           | |tick|       | |tick|            | base_assistant_engine_specification  |
+----------------+------------+-------------+-------------+------------------+--------------+-------------------+--------------------------------------+



.. |tick| unicode:: U+2705 .. tick sign
.. |cross| unicode:: U+274C .. cross sign



