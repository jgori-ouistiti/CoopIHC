from coopihc.helpers import hard_flatten

from coopihc.space.Space import Space
from coopihc.space.utils import NotASpaceError

import gym
import functools
import numpy
import warnings


class WrongConvertorWarning(Warning):
    """Warning raised when the value is not contained in the space."""

    __module__ = Warning.__module__


class Train:
    """Generic Wrapper to make bundles compatibles with gym.Env

    This is a generic Wrapper to make bundles compatibles with gym.Env. It is mainly here to be subclassed by other wrappers.

    Depending on the spaces you are using, you might need to provide a wrapper to accustom the fact that coopihc spaces can take any values whereas e.g. gym discrete spaces have to be unit-spaced values.

    .. note::

        Experimental: This class automatically build wrappers to account for the transformation between a bundle and an env, but we offer no guarantees that it will work in all cases. It might also likely be faster (computationnally) to hard code your own wrappers.


    :param bundle: bundle to wrap
    :type bundle: `Bundle<coopihc.bundle.Bundle.Bundle`
    :param train_user: whether to train the user, defaults to True
    :type train_user: bool, optional
    :param train_assistant: whether to train the assistant, defaults to True
    :type train_assistant: bool, optional
    :param API: API with which the bundle will be made compatible for, defaults to "gym-force". In gym force, a limited gym compatible environment is created, which casts everything to float32 and boxes.
    :type API: str, optional
    :param observation_dict: to filter out observations, you can apply a dictionnary, defaults to None. e.g.:

    ..code-block:: python

        filterdict = OrderedDict(
        {
            "user_state": OrderedDict({"goal": 0}),
            "task_state": OrderedDict({"x": 0}),
        }
    )

    :type observation_dict: collections.OrderedDict, optional
    :param reset_dic: During training, the bundle will be repeatedly reset. Pass the reset_dic here (see bundle reset mechanism), defaults to {}
    :type reset_dic: dict, optional
    :param reset_turn: During training, the bundle will be repeatedly reset. Pass the reset_turn here (see bundle reset_turn mechanism), defaults to 0
    :type reset_turn: int, optional
    """

    def __init__(
        self,
        bundle,
        *args,
        train_user=True,
        train_assistant=True,
        API="gym-force",
        observation_dict=None,
        reset_dic={},
        reset_turn=0,
        **kwargs
    ):

        self.bundle = bundle
        self.train_user = train_user
        self.train_assistant = train_assistant
        self.observation_dict = observation_dict
        self.reset_dic = reset_dic
        self.reset_turn = reset_turn
        self._convertor = None

        if API == "gym":
            self._convertor = GymConvertor
        elif API == "gym-force":
            self._convertor = GymForceConvertor
        else:
            raise NotImplementedError

        (
            self.action_space,
            self.action_wrappers,
        ) = self._get_action_spaces_and_wrappers()

        (
            self.observation_space,
            self.observation_wrappers,
        ) = self._get_observation_spaces_and_wrappers()

    def _get_observation_spaces_and_wrappers(self):
        """_get_observation_spaces_and_wrappers

        Get obs spaces and wrappers by querying an observation from the bundle and calling the convertor."""
        obs = self.bundle.reset()
        if self.observation_dict is None:
            filter = obs
        else:
            filter = self.observation_dict
        spaces = hard_flatten(obs.filter("spaces", filter))
        return self.convertor.get_spaces_and_wrappers(spaces, "observation")[:2]

    def _get_action_spaces_and_wrappers(self):
        """_get_action_spaces_and_wrappers [summary]

        Get action spaces and wrappers. Checks who should be trained, and calls the convertor

        """
        action_spaces = []
        if self.train_user:
            user_action_space = self.bundle.game_state["user_action"]["action"][
                "spaces"
            ]
            action_spaces.extend(user_action_space)
        else:
            user_action_space = None

        if self.train_assistant:
            assistant_action_space = self.bundle.game_state["assistant_action"][
                "action"
            ]["spaces"]
            action_spaces.extend(assistant_action_space)
        else:
            assistant_action_space = None

        self.bundle_user_action_space = user_action_space
        self.bundle_assistant_action_space = assistant_action_space

        self.convertor = self._convertor(bundle_action_spaces=action_spaces)
        return self.convertor.get_spaces_and_wrappers(action_spaces, "action")[
            :2
        ]  # no wrapper flags returned

    def _convert_observation(self, observation):
        """_convert_observation

        Hard flattens the bundle observation and casts to int or array


        """
        if isinstance(self.observation_space, gym.spaces.Discrete):
            return int(
                hard_flatten(observation.filter("values", self.observation_dict))[0]
            )
        else:
            return numpy.array(
                hard_flatten(observation.filter("values", self.observation_dict))
            )


class TrainGym(Train, gym.Env):
    """Generic Wrapper to make bundles compatibles with gym.Env

    This is a generic Wrapper to make bundles compatibles with gym.Env. Read more on the Train class.

    :param bundle: bundle to wrap
    :type bundle: `Bundle<coopihc.bundle.Bundle.Bundle`
    :param train_user: whether to train the user, defaults to True
    :type train_user: bool, optional
    :param train_assistant: whether to train the assistant, defaults to True
    :type train_assistant: bool, optional
    :param observation_dict: to filter out observations, you can apply a dictionnary, defaults to None. e.g.:

    ..code-block:: python

        filterdict = OrderedDict(
        {
            "user_state": OrderedDict({"goal": 0}),
            "task_state": OrderedDict({"x": 0}),
        }
    )

    :type observation_dict: collections.OrderedDict, optional
    :param reset_dic: During training, the bundle will be repeatedly reset. Pass the reset_dic here (see bundle reset mechanism), defaults to {}
    :type reset_dic: dict, optional
    :param reset_turn: During training, the bundle will be repeatedly reset. Pass the reset_turn here (see bundle reset_turn mechanism), defaults to 0
    :type reset_turn: int, optional
    """

    def __init__(
        self,
        bundle,
        *args,
        train_user=True,
        train_assistant=True,
        observation_dict=None,
        reset_dic={},
        reset_turn=0,
        force=False,
        **kwargs
    ):
        if force == True:
            api = "gym-force"
        else:
            api = "gym"
        super().__init__(
            bundle,
            *args,
            train_user=train_user,
            train_assistant=train_assistant,
            API=api,
            observation_dict=observation_dict,
            reset_dic=reset_dic,
            reset_turn=reset_turn,
            **kwargs,
        )

    def reset(self):
        """Reset the environment.

        :return: observation (numpy.ndarray) observation of the flattened game_state --> see gym API. rewards is a dictionnary which gives all elementary rewards for this step.

        """
        obs = self.bundle.reset(turn=self.reset_turn, dic=self.reset_dic)
        return self._convert_observation(obs)

    def step(self, action):
        """Perform a step of the environment.

        :param action: (list, numpy.ndarray) Action (or joint action for PlayBoth)

        :return: observation, reward, is_done, rewards --> see gym API. rewards is a dictionnary which gives all elementary rewards for this step.

        :meta public:
        """

        user_action = action[: len(self.bundle_user_action_space)]
        assistant_action = action[len(self.bundle_user_action_space) :]
        obs, rewards, is_done = self.bundle.step(user_action, assistant_action)

        return (
            self._convert_observation(obs),
            float(sum(rewards.values())),
            is_done,
            rewards,
        )

    def render(self, mode):
        """See Bundle and gym API

        :meta public:
        """
        self.bundle.render(mode)

    def close(self):
        """See Bundle and gym API

        :meta public:
        """
        self.bundle.close()


def partialclass(cls, *args):
    """partialclass

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
            return partialclass(
                self._ChangeLayoutAction, spaces, wrapperflag, self.bundle_action_spaces
            )

    def _get_observation_wrapper(self, spaces, wrapperflag):
        """_get_observation_wrapper [summary]

        Returns wrappers if needed


        """
        if True not in wrapperflag:
            return None
        else:
            return partialclass(self._ChangeLayoutObservation, spaces, wrapperflag)


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
        return partialclass(
            self._ChangeLayoutAction,
            spaces,
            slice_list,
            round_list,
            wflag,
            range,
            self.bundle_action_spaces,
        )

    def _get_observation_wrapper(self, spaces, slice_list, round_list, wflag, range):
        return partialclass(
            self._ChangeLayoutObservation,
            spaces,
            slice_list,
            round_list,
            wflag,
            range,
            self.bundle_action_spaces,
        )
