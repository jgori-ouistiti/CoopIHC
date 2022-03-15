from coopihc.space.Space import CatSet, Space, Interval

import numpy
import functools
import warnings
import gym
from coopihc.helpers import hard_flatten


def autospace():
    pass


def discrete_space():
    pass


def multidiscrete_space():
    pass


def continuous_space():
    pass


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
def lin_space(num=50, start=0, stop=None, endpoint=False, dtype=numpy.int64):
    if stop is None:
        stop = num + start
    return space(
        array=numpy.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype)
    )


def integer_space(N, **kwargs):
    return lin_space(num=N, **kwargs)


def box_space(high=numpy.ones((1, 1)), low=None, dtype=None):
    if low is None:
        low = -high
    return space(low=low, high=high, dtype=dtype)


def space(low=None, high=None, array=None, N=None, _function=None, **kwargs):
    if low is not None and high is not None:
        return Interval(low=low, high=high, **kwargs)
    if array is not None:
        return CatSet(array=array, **kwargs)
    if N is not None and _function is not None:
        raise NotImplementedError
    raise ValueError(
        "You have to specify either low and high, or a set, or N and function, but you provided low = {}, high = {}, set = {}, N = {}, function = {}".format(
            low, high, array, N, _function
        )
    )


def cartesian_product(*spaces):
    """cartesian_product

    Realizes the cartesian product of the spaces provided in input. For this method, continuous spaces are treated as singletons {None}.

    .. code-block:: python

        s = Space(
            numpy.array([i for i in range(3)], dtype=numpy.int16),
            "discrete",
            contains="hard",
            seed=123,
        )
        q = Space(
            [
                numpy.array([i + 6 for i in range(2)], dtype=numpy.int16),
                numpy.array([i + 6 for i in range(2)], dtype=numpy.int16),
            ],
            "multidiscrete",
            contains="hard",
            seed=789,
        )
        r = Space(
            [
                -numpy.ones((2, 2), dtype=numpy.float32),
                numpy.ones((2, 2), dtype=numpy.float32),
            ],
            "continuous",
            contains="hard",
            seed=456,
        )
        cp, shape = Space.cartesian_product(s, q, r)
        assert (
            cp
            == numpy.array(
                [
                    [0, 6, 6, None],
                    [0, 6, 7, None],
                    [0, 7, 6, None],
                    [0, 7, 7, None],
                    [1, 6, 6, None],
                    [1, 6, 7, None],
                    [1, 7, 6, None],
                    [1, 7, 7, None],
                    [2, 6, 6, None],
                    [2, 6, 7, None],
                    [2, 7, 6, None],
                    [2, 7, 7, None],
                ]
            )
        ).all()
        assert shape == [(1,), (2, 1), (2, 2)]


    :return: cartesian product and shape of associated spaces
    :rtype: tuple(numpy.ndarray, list(tuples))
    """
    arrays = []
    shape = []
    for space in spaces:
        shape.append(space.shape)
        if isinstance(space, CatSet):
            arrays.append(space.array)
        elif isinstance(
            space, Interval
        ):  # Does not deal with case when the Interval is discrete dtype.
            arrays.append(numpy.array([None]))

    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la), shape


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
            _action = []
            for a, rl in zip(action, self.round_list):
                if rl is True:
                    _action.append(numpy.round(a))
                else:
                    _action.append(a)
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
