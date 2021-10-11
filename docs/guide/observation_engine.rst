.. observation_engine:

The Observation Engines
========================
In an interactive setting, states are rarely perfectly observable by the various agents:

    * the other agent's internal states are unknown,
    * the task's state may be partially observable; e.g. a human observer is imperfect and produces noisy observations,
    * an agent's own internal state may be partially observable; e.g. a human observer with decaying memory

Furthermore, there might be a cost associated with making observations:

    * A human observer may take some time to locate a target. Usually, there will be a tradeoff between the time needed to produce and observation and the quality of that observation (speed-accuracy tradeoff)
    * A human observer may enjoy making observations which are very different (according to some criterion) to the previous ones, in which case it would be rewarded for satisfying its curiosity

*interaction-agents* provides a generic object called an observation engine which specifies how an observation is created from the game state. Each substate of the game state can be addressed specifically. For example, in the example below, the observation engine is defined in a way that it will not observe the first substate, that it will have a noisy observation of the second substate, and that it will perfectly observe the remainder of the state.

.. tikz:: The observation engine
    :include: tikz/observation_engine.tikz
    :xscale: 100
    :align: center


The ``Observation Engine`` class
---------------------------------
The Observation Engine class is the Base class from which all Observation Engines are derived. It provides nothing but a type attribute and an observe method. This method is redefined in the observation engines derived from this base class.

*interaction-agents* currently provide three observation engines:

1. ``RuleObservationEngine`` [link], which produces perfect observations on the substates targeted by the 'rule'.
2. ``NoisyRuleObservationEngine`` [link], which produces perfect and/or noisy observations on the substates targeted by a 'rule' and some 'noiserules'
3. ``ProcessObservationEngine`` [link], which produces observations by having a 'process' start from the game state and finish at another state. The intent of this engine is to use a run of a fully functioning bundle to produce observations e.g. the ``ChenEyePointingTask`` [link] from the eye module.


.. _rule-observation-engine-label:

The ``RuleObservationEngine``
---------------------------------
 A rule observation engine is initialized with a rule e.g.

.. code-block:: python

    observation_engine = RuleObservationEngine(BaseUserObservationRule)

A rule is an ``OrderedDict``, see e.g. the ``BaseUserObservationRule``

.. code-block:: python

    BaseUserObservationRule = OrderedDict([('b_state', 'all'), ('task_state', 'all'), ('user_state', 'all'), ('assistant_state', None) ])


Values in the ordered dictionnary may be

* 'all', in which case the whole substate is observed
* None, in which case none of the substate is observed
* [deprecated] a slice object, in which case the corresponding slice of the state is extracted.

Several rules are predefined:

==============================  =================  ============== ================= ====================
Rule name                           Bundle state    Task state      User state      Assistant state
==============================  =================  ============== ================= ====================
OracleObservationRule               ✔️                      ✔️              ✔️                  ✔️
BaseBlindRule                       ✔️                      ❌               ❌               ❌
TaskObservationRule                 ✔️                      ✔️              ❌               ❌
BaseUserObservationRule         ✔️                      ✔️              ✔️              ❌
BaseAssistantObservationRule        ✔️                  ✔️                 ❌                 ✔️
==============================  =================  ============== ================= ====================


.. note::

    While it may be tempting to reduce the size of the observation if some substates are not relevant, it is usually better to do this later directly from the bundle. For example, to remove the substates irrelevant for training the ``Train`` wrapper provides the ``squeeze_output()`` [link]  method.

.. note::

    TODO: this solution doesn't work anymore with slices

The ``NoisyRuleObservationEngine``
-------------------------------------

The noisy rule observation engine subclasses a rule observation engine. In addition to the 'rule', it expects a list of noiserules, which specify methods to add noise.

A noiserule is a tuple (substate, subsubstate, index, method). For example, the noiserule below specifies to apply the method numpy.random.random to ``game_state['task_state']['Targets'][0]``

.. code-block:: python

    noiserule = ('task_state', 'Targets', 0, numpy.random.random)


A noisy rule observation engine can initialized like so:

.. code-block:: python

    noiserules = [('task_state', 'Targets', 0, numpy.random.random)]
    observation_engine = NoisyRuleObservationEngine(BaseUserObservationRule, noiserules)

The ``ProcessObservationEngine``
------------------------------------
.. note::

    TODO: The intent of this engine is to use a run of a fully functioning bundle to produce observations. For example, the eye module can be used as an observation process to detect a target in a layout. The number of turns required to locate the target (i.e. time it takes to locate the target) is returned via the rewards.

This is still work in progress.
