from coopihc.space.Space import Space
import numpy
import gym
import functools


def remove_prefix(text, prefix):
    # from Python 3.9 use str.removeprefix() directly
    # copied from https://stackoverflow.com/questions/16891340/remove-a-prefix-from-a-string
    return text[text.startswith(prefix) and len(prefix) :]


class StateNotContainedWarning(Warning):
    """Warning raised when the value is not contained in the space."""

    __module__ = Warning.__module__


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


def discrete_space(possible_values, dtype=numpy.int16):
    """discrete_space

    Shortcut to generate a discrete Space object

    :param possible_values: possible values for the Space
    :type possible_values: numpy array_like
    :param dtype: type of the data, defaults to numpy.int16
    :type dtype: numpy dtype, optional
    :return: an initialized `Space<coopihc.space.Space.Space>` object
    :rtype: `Space<coopihc.space.Space.Space>`
    """
    return Space([numpy.array([possible_values], dtype=dtype)])


def continuous_space(low, high, dtype=numpy.float32):
    """continuous_space

    Shortcut to generate a continuous Space object

    :param low: lower bound
    :type low: numpy.ndarray
    :param high: upper bound
    :type high: numpy.ndarray
    :param dtype: type of the data, defaults to numpy.int16
    :type dtype: numpy dtype, optional
    :return: an initialized `Space<coopihc.space.Space.Space>` object
    :rtype: `Space<coopihc.space.Space.Space>`
    """
    return Space([low.astype(dtype), high.astype(dtype)])


def multidiscrete_space(iterable_possible_values, dtype=numpy.int16):
    """multidiscrete_space

    Shortcut to generate a multidiscrete_space Space object

    :param iterable_possible_values: list of possible values for the Space
    :type iterable_possible_values: twice iterable numpy array_like
    """
    return Space([numpy.array(i).astype(dtype) for i in iterable_possible_values])


def partialclass(cls, *args):
    class ChangeLayoutAction(cls):
        __init__ = functools.partialmethod(
            cls.__init__,
            *args,
        )

    return ChangeLayoutAction


######## ================= Below is untested code, and should replace the class below with same name.

# class _ChangeLayoutAction(gym.ActionWrapper):
#     def __init__(self, env, spaces, wrapperflag):
#         change_action = []
#         for sp, wflag in zip(spaces, wrapperflag):
#             if not wflag:
#                 change_action.append([None])
#                 continue
#             change_action.append([sp.range])

#         self.change_action = change_action
#         super().__init__(env)

#     def action(self, action):
#         _action = []
#         for n, a in enumerate(action):
#             if change_action[n] == [None]:
#                 _action.append(a)
#             else:
#                 _action.append(change_action[n][a])
#         return _action


class _ChangeLayoutAction(gym.ActionWrapper):
    def __init__(self, spaces, wrapperflag, env):
        super().__init__(env)

    def action(self, action):
        return action


class _ChangeLayoutObservation(gym.ObservationWrapper):
    def __init__(self, spaces, wrapperflag, env):
        super().__init__(env)

    def observation(self, observation):
        return observation


class RLConvertor:
    def __init__(self, interface="gym", **kwargs):
        self.interface = interface
        if self.interface != "gym":
            raise NotImplementedError
        else:
            style = kwargs.get("style")
            if style != "SB3":
                raise NotImplementedError

    def get_spaces_and_wrappers(self, spaces, mode):

        spaces, wrapperflag = self._convert(spaces)
        if mode == "action":
            wrapper = self._get_action_wrapper(spaces, wrapperflag)
        elif mode == "observation":
            wrapper = self._get_observation_wrapper(spaces, wrapperflag)
        return spaces, wrapper, wrapperflag

    def _get_info(self, spaces):
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
    def __init__(self, style="SB3"):
        super().__init__(interface="gym", style=style)

    def _convert(self, spaces):
        discrete, continuous, multidiscrete, spacewrapper_flag = self._get_info(spaces)
        # Spaces -> mix of discrete and continuous Space
        if continuous and discrete:
            raise NotImplementedError(
                "Mixing continuous and discrete Spaces requires a gym.spaces.Tuple of gym.spaces.Dict conversion, which is not implemeted yet. Consider doing it yourself an opening a PR on the project's github"
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
        if True not in wrapperflag:
            return None
        else:
            return partialclass(_ChangeLayoutAction, spaces, wrapperflag)

    def _get_observation_wrapper(self, spaces, wrapperflag):
        if True not in wrapperflag:
            return None
        else:
            return partialclass(_ChangeLayoutObservation, spaces, wrapperflag)
