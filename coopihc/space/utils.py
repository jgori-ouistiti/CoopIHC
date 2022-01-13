from coopihc.space.Space import Space

import numpy
import functools
import warnings
import gym
from coopihc.helpers import hard_flatten


# ======================== Warnings ========================
class StateNotContainedWarning(Warning):
    """Warning raised when the value is not contained in the space."""

    __module__ = Warning.__module__


class NotKnownSerializationWarning(Warning):
    """Warning raised when the State tries to serialize an item which does not have a serialize method."""

    __module__ = Warning.__module__


class ContinuousSpaceIntIndexingWarning(Warning):
    """Warning raised when the State tries to serialize an item which does not have a serialize method."""

    __module__ = Warning.__module__


class NumpyFunctionNotHandledWarning(Warning):
    """Warning raised when the numpy function is not handled yet by the StateElement."""

    __module__ = Warning.__module__


class RedefiningHandledFunctionWarning(Warning):
    """Warning raised when the numpy function is already handled by the StateElement and is going to be redefined."""

    __module__ = Warning.__module__


class WrongConvertorWarning(Warning):
    """Warning raised when the value is not contained in the space."""

    __module__ = Warning.__module__


# ======================== Errors ========================


class SpaceLengthError(Exception):
    """Error raised when the space length does not match the value length."""

    __module__ = Exception.__module__


class StateNotContainedError(Exception):
    """Error raised when the value is not contained in the space."""

    __module__ = Exception.__module__


class SpacesNotIdenticalError(Exception):
    """Error raised when the value is not contained in the space."""

    __module__ = Exception.__module__


class NotASpaceError(Exception):
    """Error raised when the object is not a space."""

    __module__ = Exception.__module__


# ======================== Shortcuts ========================


def autospace(*input_array, seed=None, contains="soft", dtype=None):
    """Wrapper to Space

    A function that makes instantiating Space less cumbersome.

    :param seed: seed for the rng (useful when sampling from the Space)
    :type seed: int, optional
    :param contains: "hard" or "soft", defaults to "soft". Changes how the Space checks whether a value belong to the space or not. See `Space<coopihc.space.Space.Space>` documentation for more information.
    :type contains: str, optional
    :param dtype: numpy dtype, defaults to None
    :type dtype: numpy.dtype, optional

    :return: A continuous, discrete, or multidiscrete space
    :rtype: `Space<coopihc.space.Space.Space>`

    Some examples:

    .. code-block:: python

        # Discrete
        assert autospace([1, 2, 3]) == Space(numpy.array([1, 2, 3]), "discrete")
        assert autospace([[1, 2, 3]]) == Space(numpy.array([1, 2, 3]), "discrete")
        assert autospace(numpy.array([1, 2, 3])) == Space(
            numpy.array([1, 2, 3]), "discrete"
        )
        assert autospace(numpy.array([[1, 2, 3]])) == Space(
            numpy.array([1, 2, 3]), "discrete"
        )
        assert autospace([numpy.array([1, 2, 3])]) == Space(
            numpy.array([1, 2, 3]), "discrete"
        )
        assert autospace([numpy.array([[1, 2, 3]])]) == Space(
            numpy.array([1, 2, 3]), "discrete"
        )


        # Multidiscrete
        assert autospace(numpy.array([[1, 2, 3], [4, 5, 6]])) == Space(
        [numpy.array([1, 2, 3]), numpy.array([4, 5, 6])], "multidiscrete"
        )
        assert autospace([1, 2, 3], [1, 2, 3, 4, 5]) == Space(
            [numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])], "multidiscrete"
        )

        assert autospace([numpy.array([1, 2, 3])], [numpy.array([1, 2, 3, 4, 5])]) == Space(
            [numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])], "multidiscrete"
        )
        assert autospace(numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])) == Space(
            [numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])], "multidiscrete"
        )
        assert autospace([numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])]) == Space(
            [numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])], "multidiscrete"
        )
        assert autospace(
            [numpy.array([[1, 2, 3]]), numpy.array([[1, 2, 3, 4, 5]])]
        ) == Space([numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])], "multidiscrete")


        # Continuous
        assert autospace(
            -numpy.array([[1, 1], [1, 1]]), numpy.array([[1, 1], [1, 1]])
        ) == Space([-numpy.ones((2, 2)), numpy.ones((2, 2))], "continuous")
        assert autospace(
            [-numpy.array([[1, 1], [1, 1]]), numpy.array([[1, 1], [1, 1]])]
        ) == Space([-numpy.ones((2, 2)), numpy.ones((2, 2))], "continuous")
        assert autospace([[-1, -1], [-1, -1]], [[1, 1], [1, 1]]) == Space(
            [-numpy.ones((2, 2)), numpy.ones((2, 2))], "continuous"
        )
        assert autospace([[[-1, -1], [-1, -1]], [[1, 1], [1, 1]]]) == Space(
            [-numpy.ones((2, 2)), numpy.ones((2, 2))], "continuous"
        )
    """
    k = 0
    while k < 5:
        k += 1
        if k == 5:
            raise AttributeError(
                "Input could not be interpreted by autospace. Please conform to one of the expected input forms"
            )
        # autospace(numpy.array(XXX))
        if len(input_array) == 1 and isinstance(input_array[0], numpy.ndarray):
            # autospace(numpy.array(1))
            if len(input_array[0].shape) == 0:
                raise AttributeError(
                    "Input array {} should be of dim > 0, but has shape {}".format(
                        input_array[0], input_array[0].shape
                    )
                )
            # autospace(numpy.array([1,2,3]))
            elif len(input_array[0].shape) == 1:
                if dtype is None:
                    dtype = numpy.int16
                return Space(
                    input_array[0].astype(dtype),
                    "discrete",
                    seed=seed,
                    contains=contains,
                )

            # autospace(numpy.array([[1,2,3]]))
            # autospace(numpy.array([[1,2,3],[1,2,3]]))
            elif len(input_array[0].shape) == 2:
                if min(input_array[0].shape) == 1:  # discrete
                    if dtype is None:
                        dtype = numpy.int16
                    return Space(
                        input_array[0].squeeze().astype(dtype),
                        "discrete",
                        seed=seed,
                        contains=contains,
                    )
                else:  # multidiscrete
                    if dtype is None:
                        dtype = numpy.int16
                    return Space(
                        [i.astype(dtype) for i in input_array[0]],
                        "multidiscrete",
                        seed=seed,
                        contains=contains,
                    )
        # autospace([X])
        elif len(input_array) == 1 and isinstance(input_array[0], list):
            input_array = input_array[0]
            # autospace([1,2,3])
            if isinstance(input_array[0], (float, int, numpy.number)):  # discrete
                if dtype is None:
                    dtype = numpy.int16
                return Space(
                    numpy.array(input_array, dtype=dtype),
                    "discrete",
                    seed=seed,
                    contains=contains,
                )
            # autospace([[1,2,3]])
            elif isinstance(input_array[0], list):
                continue
            # autospace([numpy.array(X)])
            elif isinstance(input_array[0], numpy.ndarray):
                continue
        # autospace(X, Y)
        elif len(input_array) == 2:
            low, high = input_array
            # autospace(array(X), array(Y))
            if isinstance(low, numpy.ndarray):
                if len(low.shape) == 2:
                    if dtype is None:
                        dtype = numpy.float32
                    return Space(
                        [low.astype(dtype), high.astype(dtype)],
                        "continuous",
                        seed=seed,
                        contains=contains,
                    )
                else:
                    if dtype is None:
                        dtype = numpy.int16
                    return Space(
                        [low.astype(dtype).squeeze(), high.astype(dtype).squeeze()],
                        "multidiscrete",
                        seed=seed,
                        contains=contains,
                    )
            elif isinstance(low, list):
                if isinstance(low[0], (float, int, numpy.number)):
                    if dtype is None:
                        dtype = numpy.int16
                    return Space(
                        [numpy.array(low, dtype=dtype), numpy.array(high, dtype=dtype)],
                        "multidiscrete",
                        seed=seed,
                        contains=contains,
                    )
                elif isinstance(low[0], numpy.ndarray):
                    if dtype is None:
                        dtype = numpy.int16
                    return Space(
                        [low[0].astype(dtype), high[0].astype(dtype)],
                        "multidiscrete",
                        seed=seed,
                        contains=contains,
                    )
                elif isinstance(low[0], list):
                    if dtype is None:
                        dtype = numpy.float32
                    return Space(
                        [numpy.array(low, dtype=dtype), numpy.array(high, dtype=dtype)],
                        "continuous",
                        seed=seed,
                        contains=contains,
                    )
                else:
                    raise NotImplementedError
        # autospace(X, Y, Z)
        elif len(input_array) > 2:
            _arr_sample = input_array[0]
            if isinstance(_arr_sample, numpy.ndarray):
                if dtype is None:
                    dtype = numpy.int16
                return Space(
                    [_arr.astype(dtype).squeeze() for _arr in input_array],
                    "multidiscrete",
                    seed=seed,
                    contains=contains,
                )
            elif isinstance(_arr_sample, list):
                if isinstance(_arr_sample[0], (float, int, numpy.number)):
                    if dtype is None:
                        dtype = numpy.int16
                    return Space(
                        [numpy.array(_arr, dtype=dtype) for _arr in input_array],
                        "multidiscrete",
                        seed=seed,
                        contains=contains,
                    )
                elif isinstance(_arr_sample[0], numpy.ndarray):
                    if dtype is None:
                        dtype = numpy.int16
                    return Space(
                        [_arr.astype(dtype) for _arr in input_array],
                        "multidiscrete",
                        seed=seed,
                        contains=contains,
                    )


def discrete_space(array, dtype=numpy.int16, **kwargs):
    """discrete

    Shortcut. If not successful, forwards to autospace

    .. code-block:: python

        discrete_space(numpy.array([1,2,3]))

    :param array: 1d numpy array
    :type array: numpy.ndarray
    :return: discrete Space
    :rtype: `Space<coopihc.space.Space.Space>`
    """
    try:
        return Space(array, "discrete", dtype=dtype, **kwargs)
    except:
        return autospace(array, dtype=dtype, **kwargs)


def continuous_space(low, high, dtype=numpy.float32, **kwargs):
    """continuous

    Shortcut. If not successful, forwards to autospace

    .. code-block:: python

        continuous_space(-numpy.ones((2,2)), numpy.ones((2,2)))

    :param low: 2d numpy array
    :type low: numpy.ndarray
    :param high: 2d numpy array
    :type high: numpy.ndarray
    :return: continuous Space
    :rtype: `Space<coopihc.space.Space.Space>`
    """
    try:
        return Space([low, high], "continuous", dtype=dtype, **kwargs)
    except:
        return autospace(low, high, dtype=dtype, **kwargs)


def multidiscrete_space(array_list, dtype=numpy.int16, **kwargs):
    """multidiscrete

    Shortcut. If not successful, forwards to autospace

    .. code-block:: python

        multidiscrete_space([numpy.array([1,2,3]), numpy.array([1,2,3,4,5])])


    :param array_list: list of 1d numpy arrays
    :type array_list: list
    :return: multidiscrete Space
    :rtype: `Space<coopihc.space.Space.Space>`
    """
    try:
        return Space(array_list, "multidiscrete", dtype=dtype, **kwargs)
    except:
        return autospace(array_list, dtype=dtype, **kwargs)


# ======================== Convertors ========================


def _partialclass(cls, *args):
    """_partialclass

    Initializes a class by pre-passing it some arguments. Same as functools.partial applied to class inits.


    """

    class ChangeLayoutAction(cls):
        __init__ = functools.partialmethod(
            cls.__init__,
            *args,
        )

    return ChangeLayoutAction


class RLConvertor:
    """Help convert Bundle to RL Envs

    Helper class to convert Bundles to Reinforcement Learning environments

    :param interface: name of the API to which bundles is converted to, defaults to "gym"
    :type interface: str, optional
    :param bundle_action_spaces: the bundle action_spaces, defaults to None
    :type bundle_action_spaces: `State<coopihc.space.State.State>`, optional
    """

    def __init__(self, interface="gym", bundle_action_spaces=None, **kwargs):
        self.interface = interface
        if self.interface != "gym":
            raise NotImplementedError
        else:
            style = kwargs.get("style")
            if style != "SB3":
                raise NotImplementedError
        self.bundle_action_spaces = bundle_action_spaces

    def get_spaces_and_wrappers(self, spaces, mode):
        """get_spaces_and_wrappers

        Main method to call.

        Return converted spaces, a potential wrapper and wrapper flags (True if space needs wrapper component). Currently, the only space that needs a wrapper potentially is DIscrete Space (when the start action is not 0 and actions not equally unit-spaced)

        :param spaces: bundle spaces
        :type spaces: list(Spaces`coopihc.spaces.Space.Space`)
        :param mode: 'action' or 'observation'
        :type mode: string
        :return: spaces, wrapper, wrapperflag
        :rtype: tuple([gym.spaces], [gym.env wrapper], [bool])
        """
        spaces, *gawargs = self._convert(spaces)
        if mode == "action":
            wrapper = self._get_action_wrapper(spaces, *gawargs)
        elif mode == "observation":
            wrapper = self._get_observation_wrapper(spaces, *gawargs)
        return spaces, wrapper, *gawargs

    def _get_info(self, spaces):
        """_get_info

        gather intel on bundle spaces, mainly whether the spaces are continuous or discrete.

        :param spaces: bundle spaces
        :type spaces: list(Spaces`coopihc.spaces.Space.Space`)

        """
        # Check what mix spaces is
        discrete = False
        continuous = False
        multidiscrete = 0
        for space in spaces:
            if space.continuous:
                continuous = True
            else:
                discrete = True
                multidiscrete += len(space)
            if not isinstance(space, Space):
                raise NotASpaceError(
                    "All items in spaces should be of type Space, but space is of type {} ".format(
                        type(space).__name__
                    )
                )
        if continuous and discrete:
            return True, True, None, None
        # Spaces -> collection of Discrete Space
        if discrete:
            # Check which spaces will need an actionwrapper
            spacewrapper_flag = []
            for space in spaces:
                if multidiscrete > 1:
                    for _space in space:
                        if not (
                            _space.range[0].squeeze().tolist()[0] == 0
                            and numpy.mean(  # From gym 0.21, there is an option to offset the start, but only for discrete
                                numpy.diff(_space.range)
                            )
                            == 1.0
                            and numpy.std(numpy.diff(_space.range)) == 0.0
                        ):
                            spacewrapper_flag.append(True)
                        else:
                            spacewrapper_flag.append(False)
                else:
                    _space = space
                    if not (
                        _space.range[0].squeeze().tolist()[0] == 0
                        and numpy.mean(  # From gym 0.21, there is an option to offset the start, but only for discrete
                            numpy.diff(_space.range)
                        )
                        == 1.0
                        and numpy.std(numpy.diff(_space.range)) == 0.0
                    ):
                        spacewrapper_flag.append(True)
                    else:
                        spacewrapper_flag.append(False)
        else:
            spacewrapper_flag = [False for space in spaces]

        return discrete, continuous, multidiscrete, spacewrapper_flag


class GymConvertor(RLConvertor):
    """Help convert Bundle to RL Gym Envs

    Helper class to convert Bundles to Gym Reinforcement Learning environments, with a certain library in mind. (Not all libraries may support all action and observation spaces.) Works only for fully discrete or continuous spaces.

    :param style: lib name
    :type interface: str, optional
    """

    def __init__(self, style="SB3", **kwargs):
        super().__init__(interface="gym", style=style, **kwargs)

    class _ChangeLayoutAction(gym.ActionWrapper):
        """_ChangeLayoutAction

        Class prototype

        :param gym: [description]
        :type gym: [type]
        """

        def __init__(self, spaces, wrapperflag, bundle_action_spaces, env):
            super().__init__(env)
            change_action = []
            for bundle_space, wflag in zip(bundle_action_spaces, wrapperflag):
                if not wflag:
                    change_action.append(None)
                    continue
                change_action.append(bundle_space.range[0])
            self.change_action = change_action

        def action(self, action):
            _action = []
            for n, a in enumerate(action):
                if not ((self.change_action[n]).any()):
                    _action.append(a)
                else:
                    _action.append(self.change_action[n].squeeze()[a])
            return _action

    # Leacing this at prototype, because in practical cases its useless to have observation wrappers for training.
    class _ChangeLayoutObservation(gym.ObservationWrapper):
        """_ChangeLayoutObservation

        Class prototype --> In practice is useless because further observation wrappers will further be applied to normalize the input to the NN.

        :param gym: [description]
        :type gym: [type]
        """

        def __init__(self, spaces, wrapperflag, env):
            super().__init__(env)

        def observation(self, observation):
            return observation

    def _convert(self, spaces):
        """_convert

        Convert bundle spaces to gym spaces

        :param spaces: [description]
        :type spaces: [type]
        :raises NotImplementedError: [description]
        :raises NotImplementedError: [description]
        :return: [description]
        :rtype: [type]
        """
        discrete, continuous, multidiscrete, spacewrapper_flag = self._get_info(spaces)
        # Spaces -> mix of discrete and continuous Space
        if continuous and discrete:
            raise NotImplementedError(
                "This Convertor does not deal with mixed continuous and discrete spaces, because this requires gym.spaces.Tuple or gym.spaces.Dict support, which is currently not the case. Try using GymForceConvertor instead."
            )
        if discrete:
            # Deal with one-unit collections
            if multidiscrete == 1:
                return gym.spaces.Discrete(spaces[0].N), spacewrapper_flag
            else:
                if len(spaces) == 1:
                    return (
                        gym.spaces.MultiDiscrete([s.N for s in spaces[0]]),
                        spacewrapper_flag,
                    )
                else:
                    return (
                        gym.spaces.MultiDiscrete([s.N for s in spaces]),
                        spacewrapper_flag,
                    )
        # Spaces -> collection of continuous Space
        if continuous:
            if len(spaces) != 1:
                raise NotImplementedError(
                    "Mixing continuous spaces requires a gym.spaces.Tuple of gym.spaces.Dict conversion, which is not implemeted yet. Consider doing it yourself an opening a PR on the project's github"
                )
            else:
                return gym.spaces.Box(
                    spaces[0].low,
                    spaces[0].high,
                    dtype=spaces[0].dtype,
                    # seed=spaces[0].seed,   Starting from some gym version, this is possible
                ), [False]

    def _get_action_wrapper(self, spaces, wrapperflag):
        """_get_action_wrapper

        Returns wrappers if needed


        """
        if True not in wrapperflag:
            return None
        else:
            return _partialclass(
                self._ChangeLayoutAction, spaces, wrapperflag, self.bundle_action_spaces
            )

    def _get_observation_wrapper(self, spaces, wrapperflag):
        """_get_observation_wrapper [summary]

        Returns wrappers if needed


        """
        if True not in wrapperflag:
            return None
        else:
            return _partialclass(self._ChangeLayoutObservation, spaces, wrapperflag)


class WrongConvertorError(Exception):
    """WrongConvertorError

    Raised if the current convertor can't handle the space to convert.

    """

    __module__ = Exception.__module__


class GymForceConvertor(RLConvertor):
    """GymForceConvertor

    Helper class to convert Bundles to Gym Reinforcement Learning environments, with a certain library in mind. Casts all the spaces to Boxes, which forces the compatibility with SB3.

    :param style: RL-library, defaults to "SB3" (stables_baselines3)
    :type style: str, optional
    """

    def __init__(self, style="SB3", **kwargs):

        super().__init__(interface="gym", style=style, **kwargs)

    class _ChangeLayoutAction(gym.ActionWrapper):
        """_ChangeLayoutAction

        Class prototype

        """

        def __init__(
            self,
            spaces,
            slice_list,
            round_list,
            wflag,
            range,
            bundle_action_spaces,
            env,
        ):
            self.round_list = round_list
            super().__init__(env)

        def action(self, action):
            print("\n====AW")
            print(action)
            print(self.round_list)
            _action = []
            for a, rl in zip(action, self.round_list):
                if rl is True:
                    _action.append(numpy.round(a))
                else:
                    _action.append(a)
            print(_action)
            return _action

    class _ChangeLayoutObservation(gym.ObservationWrapper):
        """_ChangeLayoutObservation

        Class prototype

        """

        def __init__(
            self,
            spaces,
            slice_list,
            round_list,
            continuous,
            range,
            bundle_action_spaces,
            env,
        ):
            super().__init__(env)

        def observation(self, observation):
            return observation

        def action(self, action):
            _action = []
            for n, a in enumerate(action):
                if not ((self.change_action[n]).any()):
                    _action.append(a)
                else:
                    _action.append(self.change_action[n].squeeze()[a])
            return _action

    def _convert(self, spaces):
        """_convert

        Convert the bundle spaces to gym spaces and output some needed statistics
        """
        discrete, continuous, multidiscrete, spacewrapper_flag = self._get_info(spaces)
        space_list = None
        # Spaces -> mix of discrete and continuous Space
        if not (continuous and discrete):
            warnings.warn(
                "This Convertor is meant to deal with mixed continuous and discrete spaces only, because it will cast discrete spaces to boxes. Try using GymConvertor instead, by passing force=False to TrainGym ",
                WrongConvertorWarning,
            )
        lower_bound = []
        upper_bound = []
        slice_list = []
        round_list = []
        wflag = []
        _range = []
        k = 0
        for space in spaces:
            # Continuous
            if space.continuous:
                lower_bound.append(hard_flatten(space.low))
                upper_bound.append(hard_flatten(space.high))
                slice_list.append(slice(k, k + numpy.prod(space.shape)))
                print(round_list)
                print([False for i in range(numpy.prod(space.shape))])
                round_list.extend([False for i in range(numpy.prod(space.shape))])

                k += numpy.prod(space.shape)
                wflag.append(False)
                _range.append([])
            # Multidiscrete
            elif not space.continuous and len(space) > 1:
                lower_bound.append([numpy.min(sr) for sr in space.range])
                upper_bound.append([numpy.max(sr) for sr in space.range])
                slice_list.append(slice(k, k + len(space)))
                round_list.extend([True for i in range(len(space))])
                k += len(space)
                wflag.append(True)
                _range.append(space.range)
            # Discrete
            elif not space.continuous and len(space) == 1:
                lower_bound.append(numpy.min(space.range))
                upper_bound.append(numpy.max(space.range))
                slice_list.append(slice(k, k + 1))
                round_list.extend([True])
                k += 1
                wflag.append(True)
                _range.append(space.range)
        return (
            gym.spaces.Box(
                low=numpy.array(hard_flatten(lower_bound)).reshape((-1,)),
                high=numpy.array(hard_flatten(upper_bound)).reshape((-1,)),
            ),
            slice_list,
            round_list,
            wflag,
            _range,
        )

    def _get_action_wrapper(self, spaces, slice_list, round_list, wflag, range):
        return _partialclass(
            self._ChangeLayoutAction,
            spaces,
            slice_list,
            round_list,
            wflag,
            range,
            self.bundle_action_spaces,
        )

    def _get_observation_wrapper(self, spaces, slice_list, round_list, wflag, range):
        return _partialclass(
            self._ChangeLayoutObservation,
            spaces,
            slice_list,
            round_list,
            wflag,
            range,
            self.bundle_action_spaces,
        )


def example_state():
    from coopihc.space.State import State
    from coopihc.space.StateElement import StateElement

    state = State()
    substate = State()
    substate["x1"] = StateElement(1, discrete_space([1, 2, 3]))
    substate["x2"] = StateElement(
        [1, 2, 3],
        multidiscrete_space(
            [
                [0, 1, 2],
                [1, 2, 3],
                [
                    0,
                    1,
                    2,
                    3,
                ],
            ]
        ),
    )
    substate["x3"] = StateElement(
        1.5 * numpy.ones((3, 3)),
        continuous_space(numpy.ones((3, 3)), 2 * numpy.ones((3, 3))),
    )

    substate2 = State()
    substate2["y1"] = StateElement(1, discrete_space([1, 2, 3]))
    substate2["y2"] = StateElement(
        [1, 2, 3],
        multidiscrete_space(
            [
                [0, 1, 2],
                [1, 2, 3],
                [
                    0,
                    1,
                    2,
                    3,
                ],
            ]
        ),
    )
    state["sub1"] = substate
    state["sub2"] = substate2
    return state


def example_game_state():
    from coopihc.space.State import State
    from coopihc.space.StateElement import StateElement

    return State(
        game_info=State(
            turn_index=StateElement(
                numpy.array([0]), autospace([0, 1, 2, 3]), out_of_bounds_mode="raw"
            ),
            round_index=StateElement(
                numpy.array([1]), autospace([0, 1]), out_of_bounds_mode="raw"
            ),
        ),
        task_state=State(
            position=StateElement(
                2, autospace([0, 1, 2, 3]), out_of_bounds_mode="clip"
            ),
            targets=StateElement(
                [0, 1],
                autospace([0, 1, 2, 3], [0, 1, 2, 3]),
                out_of_bounds_mode="warning",
            ),
        ),
        user_state=State(
            goal=StateElement(0, autospace([0, 1, 2, 3]), out_of_bounds_mode="warning")
        ),
        assistant_state=State(
            beliefs=StateElement(
                numpy.array(
                    [
                        [0.125],
                        [0.125],
                        [0.125],
                        [0.125],
                        [0.125],
                        [0.125],
                        [0.125],
                        [0.125],
                    ]
                ),
                autospace(
                    [
                        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                    ]
                ),
                out_of_bounds_mode="error",
            )
        ),
        user_action=State(
            action=StateElement(1, autospace([-1, 0, 1]), out_of_bounds_mode="warning")
        ),
        assistant_action=State(
            action=StateElement(2, autospace([0, 1, 2, 3]), out_of_bounds_mode="error")
        ),
    )
