.. modularity:

Modularity
===============

The API used in *CoopIHC* being straightforward, one can create new classes from scratch, by subclassing one of the core classes, adding a few methods, and adhering to a few conventions.

Another way in which one can create new classes, which we explain now, is by re-using existing classes, and wrapping them appropriately.
We present a fully worked example where a complex sequence of interactions is described in a relatively simple manner using relatively few lines of code, and by maximally re-using existing code.



Description of the example
------------------------------

In the :doc:`quickstart <quickstart>`, we presented a BIG assistant which would assist a user in selecting a target. While proving more efficient than the unassisted version, that evaluation might have been biased: since the user cannot predict the next position of the cursor when using the BIG assistant, it needs to locate the target which is costly. This has not been accounted for. In this example, we re-use an existing bundle that was made to model eye movements, and incorporate it to our user model, to simulate the time it takes to find the cursor.


Eye-movement model
--------------------

The eye-movement model is explained in the :doc:`User Modeling Failitation <user_modeling>` page. It is loaded simply by the following lines:

.. code-block:: python

    fitts_W = 4e-2
    fitts_D = 0.8
    perceptualnoise = 0.09
    oculomotornoise = 0.09
    task = ChenEyePointingTask(fitts_W, fitts_D, dimension = 1)
    user = ChenEye( perceptualnoise, oculomotornoise, dimension = 1)
    obs_bundle = SinglePlayUserAuto(task, user, start_at_action = True)


The task that is solved by this bundle is to position the eye (fixation) on top of the target. The user is in charge of choosing the next fixation, based on noisy information it gets from the target location. We will consider that the target is the cursor, while the initial eye fixation is the last position of the cursor. We will then let the bundle play out in time, finding the cursor in a given number of steps.


This bundle, like any other, can be reset to a given state via a reset dictionary (see :doc:`Bundles<../modules/core/bundle>`), for example


.. code-block:: python

    reset_dic = {'task_state':
                                {   'Targets': .5,
                                    'Fixation': -.5    }
                            }

    obs_bundle.reset(reset_dic)
    >>> print(obs_bundle.game_state)
      Index  Label                      Value     Space        Possible Value
    -------  -------------------------  --------  -----------  ----------------
          0  turn_index|0               0         Discrete(2)  None
          1  task_state|Targets|0       [0.5]     Box(1,)      None
          2  task_state|Fixation|0      [-0.5]    Box(1,)      None
          3  user_state|belief|0    [0]       Box(1,)      [None]
          4  user_state|belief|1    [[1000]]  Box(1, 1)    [None]
          5  user_action|action|0   None      Box(1,)      None
          6  assistant_action|action|0  None      None         [None]

Adapting the task
----------------------


To simulate tracking the cursor, we can reset the bundle by passing the new cursor location as a target and the old cursor location as the current fixation. Then, playing the bundle will result in several steps until the fixation reaches the target. To do so, we need to have the old position of the cursor as part of the task state. Let us subclass the existing pointing task described in :doc:`the quickstart <quickstart>` to add an ``OldPosition`` state:

.. code-block:: python

    class OldPositionMemorizedSimplePointingTask(SimplePointingTask):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.memorized = None

        def reset(self, reset_dic = None):
            super().reset(reset_dic)
            self.state['OldPosition'] = copy.deepcopy(self.state['Position'])

        def user_step(self, *args, **kwargs):
            self.memorized = copy.deepcopy(self.state['Position'])
            obs, rewards, is_done, _doc = super().user_step(*args, **kwargs)
            obs['OldPosition'] = self.memorized
            return obs, rewards, is_done, _doc

        def assistant_step(self, *args, **kwargs):
            self.memorized = copy.deepcopy(self.state['Position'])
            obs, rewards, is_done, _doc = super().assistant_step(*args, **kwargs)
            obs['OldPosition'] = self.memorized
            return obs, rewards, is_done, _doc


    pointing_task = OldPositionMemorizedSimplePointingTask(gridsize = 31, number_of_targets = 8, mode = 'position')
    bundle = _DevelopTask(pointing_task)
    bundle.reset()
    >>> print(bundle.game_state)
      Index  Label                      Value    Space         Possible Value
    -------  -------------------------  -------  ------------  ----------------
          0  turn_index|0               0        Discrete(2)   None
          1  task_state|Position|0      18       Discrete(31)  [None]
          2  task_state|Targets|0       7        Discrete(31)  [None]
          3  task_state|Targets|1       9        Discrete(31)  [None]
          4  task_state|Targets|2       10       Discrete(31)  [None]
          5  task_state|Targets|3       12       Discrete(31)  [None]
          6  task_state|Targets|4       16       Discrete(31)  [None]
          7  task_state|Targets|5       17       Discrete(31)  [None]
          8  task_state|Targets|6       19       Discrete(31)  [None]
          9  task_state|Targets|7       26       Discrete(31)  [None]
         10  task_state|OldPosition|0   18       Discrete(31)  [None]
         11  user_action|action|0   None     None          [None]
         12  assistant_action|action|0  None     None          [None]


Our custom observation Engine
--------------------------------

We can now wrap our bundle for the eye-movement model into an observation engine. First we must notice that the states are not compatible: the eye-movement model is expressed in a [-1,1] Box, while the pointing model is in a {0,1,2,...,29,30} grid. The :doc:`StateElement<../modules/core/space>` object has a ``cast`` method that allows one to cast states from one space to another, see e.g. below

.. code-block:: python

    x = StateElement(   values = [4],
            spaces = [gym.spaces.Discrete(9)],
            possible_values = [[None]])

    y = StateElement(   values = [None],
                    spaces = [gym.spaces.Box(-1, 1, shape = (1,))],
                    possible_values = [None]
                    )

    ret = x.cast(y, inplace = False)
    print(ret)
    >>> 
    value:	[array([0.], dtype=float32)]
    spaces:	[Box(1,)]
    possible values:	[None]


Casting can be done in place or not, and works from several spaces to several other spaces, see :doc:`StateElement<states>` for more information.



We are now set to wrap the bundle into an observation engine. To do so, we simply define an observe method, which does the following:

* Gets the current an old cursor positions and casts them to targets and fixations.
* reset the observation bundle so that targets and fixations match the cursor positions.
* Let the bundle play, collect rewards
* cast the fixation and targets back to cursor positions.
* return the new state and rewards

.. code-block:: python

    class ChenEyeObservationEngineWrapper(ObservationEngine):

        def __init__(self, obs_bundle):
            super().__init__()
            self.type = 'process'
            self.obs_bundle = obs_bundle
            self.obs_bundle.reset()

        def observe(self, game_state):
            # Cast to the box of the obs bundle
            target = game_state['task_state']['Position'].cast(self.obs_bundle.game_state['task_state']['Targets'], inplace = False)
            fixation = game_state['task_state']['OldPosition'].cast(self.obs_bundle.game_state['task_state']['Fixation'], inplace = False)
            reset_dic = {'task_state':
                            {   'Targets': target,
                                'Fixation': fixation    }
                        }

            self.obs_bundle.reset(reset_dic)
            is_done = False
            rewards = 0
            while True:
                obs, reward, is_done, _doc = self.obs_bundle.step()
                rewards += reward
                if is_done:
                    break
            obs['task_state']['Fixation'].cast(game_state['task_state']['OldPosition'], inplace = True)
            obs['task_state']['Targets'].cast(game_state['task_state']['Position'], inplace = True)
            return game_state, rewards


Cascading Observation Engines
----------------------------------

This observation engine can now be used by an agent. Now, it might be that different bundles be used to produce an observation, e.g. if I want to add noise to some other substate. Several observation engines can be combined via the ``CascadedObservationEngine``. Below, we combine our newly defined observation engine with the original one:

.. code-block:: python

    cursor_tracker = ChenEyeObservationEngineWrapper(obs_bundle)
    base_user_engine_specification  =    [ ('turn_index', 'all'),
                                        ('task_state', 'all'),
                                        ('user_state', 'all'),
                                        ('assistant_state', None),
                                        ('user_action', 'all'),
                                        ('assistant_action', 'all')
                                        ]
    default_observation_engine = RuleObservationEngine(
            deterministic_specification = base_user_engine_specification,
            )

    observation_engine = CascadedObservationEngine([cursor_tracker, default_observation_engine])

With ``CascadedObservationEngine``, each observation engine is applied in the order it is mentioned in the list. Here, the observation will first be produced by ``cursor_tracker``. That observation will then be passed to ``default_observation_engine``, which will return the true final observation used by the agent.

Now, simply continue as usual, e.g. to evaluate the setup:

.. code-block:: python

    binary_user = CarefulPointer(observation_engine = observation_engine)
    BIGpointer = BIGGain()


    bundle = PlayNone(pointing_task, binary_user, BIGpointer)
    game_state = bundle.reset()
    bundle.render('plotext')
    rewards = []
    while True:
    reward, is_done, reward_list = bundle.step()
    rewards.append(reward_list)
    bundle.render('plotext')
    if is_done:
        break

The full code for this example is found :download:`here<code/modularity.py>`
