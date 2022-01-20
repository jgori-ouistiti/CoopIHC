.. wrappers:

Wrappers
==================

.. warning::

    outdated

Wrappers are used to transform a bundle, in the spirit of gym wrappers.

List of Wrappers (ongoing):
-----------------------------

* ``Train`` [link]. Use this class to wrap a bundle up, so that it is compatible with the gym API and can be trained with off-the-shelf RL algorithms. A bundle wrapped in ``Train`` is called an environment (env).

    .. code-block:: python

        bundle = SinglePlayUser(task, user)
        env = Train(bundle)

* to be completed


The Train wrapper
-------------------

This wrapper has two goals

1. Make the input and outputs and API calls of bundle compatible with gym.Env
2. Enhance training, by providing some transformations on input and output data.

The first goal is achieved by flattening  the bundle.step() output (a nested OrderedDict) to a single dimensional array, whose components are listed in the order of the game state.

For the second goal, Train provides:

1. The ``env.squeeze_output(extract_object)`` method, which reduces the observation by removing irrelevant substates of the game state, from the flattened output

    .. note::

        Using ``print(bundle)`` will show the flattened game state and its corresponding labels and indices.

    .. code-block:: python

        bundle = SinglePlayUser(task, user)
        >>> print(bundle)
        b_state/next_agent/0  0
        task_state/Targets/0  -0.443
        task_state/Targets/1  0.232
        user_state/MuBelief/0  -0.230
        user_state/MuBelief/1  0.240
        user_state/SigmaBelief/0  0.016
        user_state/SigmaBelief/1  -0.000
        user_state/SigmaBelief/2  -0.000
        user_state/SigmaBelief/3  0.018
        user_state/Fixation/0  -0.045
        user_state/Fixation/1  0.505
        user_state/AssistantAction/0  -0.932
        assistant_state/UserAction/0  0.866
        assistant_state/UserAction/1  0.460

    Let us say we want to train using only the information from the two MuBelief components, then we would do for example:

    .. code-block:: python

        env = Train(bundle)
        env.squeeze_output(slice(3,5,1))
        >>> print(env.observation_space)
        Box(2,)


    .. note::

        ``extract_object`` can be any object that is valid for fancy indexing e.g. slices, arrays.



2. TODO: normalization of outputs and inputs to [-1,1] --> gym wrappers should be compatible, verify this
3. TODO: The action space flattening to box used in the bundle should be moved to the train wrapper.
