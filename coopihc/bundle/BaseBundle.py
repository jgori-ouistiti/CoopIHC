from random import random
from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.base.elements import discrete_array_element, cat_element

import numpy
import yaml
import matplotlib.pyplot as plt
import copy


class BaseBundle:
    """Main class for bundles.

    Main class for bundles. This class is subclassed by Bundle, which defines the interface with which to interact.

    A bundle combines a task with a user and an assistant. The bundle creates the ``game_state`` by combining the task, user and assistant states with the turn index and both agent's actions.

    The bundle takes care of all the messaging between classes, making sure the gamestate and all individual states are synchronized at all times.

    The bundle implements a forced reset mechanism, where each state of the bundle can be forced to a particular state via a dictionnary mechanism (see :py:func:reset)

    The bundle also takes care of rendering each of the three component in a single place.


    :param task: (:py:class:`coopihc.interactiontask.InteractionTask.InteractionTask`) A task that inherits from ``InteractionTask``
    :param user: (:py:class:`coopihc.agents.BaseAgent.BaseAgent`) a user which inherits from ``BaseAgent``
    :param assistant: (:py:class:`coopihc.agents.BaseAgent.BaseAgent`) an assistant which inherits from ``BaseAgent``

    :meta public:
    """

    turn_dict = {
        "after_assistant_action": 0,
        "before_user_action": 1,
        "after_user_action": 2,
        "before_assistant_action": 3,
    }

    def __init__(
        self,
        task,
        user,
        assistant,
        *args,
        reset_random=False,
        reset_start_after=-1,
        reset_go_to=0,
        **kwargs,
    ):
        self._reset_random = reset_random
        self._reset_start_after = reset_start_after
        self._reset_go_to = reset_go_to

        self.kwargs = kwargs
        self.task = task
        self.task.bundle = self
        self.user = user
        self.user.bundle = self
        self.assistant = assistant
        self.assistant.bundle = self

        # Form complete game state
        self.game_state = State()

        turn_index = cat_element(
            N=4, init=0, out_of_bounds_mode="raw", dtype=numpy.int8
        )
        round_index = discrete_array_element(
            init=0, low=0, high=numpy.iinfo(numpy.int64).max, out_of_bounds_mode="raw"
        )

        self.game_state["game_info"] = State()
        self.game_state["game_info"]["turn_index"] = turn_index
        self.game_state["game_info"]["round_index"] = round_index
        self.game_state["task_state"] = task.state
        self.game_state["user_state"] = user.state
        self.game_state["assistant_state"] = assistant.state

        # here there is a small caveat: you can not access action states in the game_state at finit, you have to pass through the agent instead. This is due to the current way of creating the game_state.

        self.task.finit()
        self.user.finit()
        self.assistant.finit()

        if user.policy is not None:
            self.game_state["user_action"] = user.policy.action_state
        else:
            self.game_state["user_action"] = State()
            self.game_state["user_action"]["action"] = array_element()
        if assistant.policy is not None:
            self.game_state["assistant_action"] = assistant.policy.action_state
        else:
            self.game_state["assistant_action"] = State()
            self.game_state["assistant_action"]["action"] = array_element()

        # This will not work sometimes

        # self.task.finit()
        # self.user.finit()
        # self.assistant.finit()

        # Needed for render
        self.active_render_figure = None
        self.figure_layout = [211, 223, 224]
        self.rendered_mode = None
        self.render_perm = False
        self.playspeed = 0.1

    def __repr__(self):
        """__repr__

        Pretty representation for Bundles.

        :return: pretty bundle print
        :rtype: string
        """
        return "{}\n".format(self.__class__.__name__) + yaml.safe_dump(
            self.__content__()
        )

    def __content__(self):
        """__content__

        Custom class representation

        :return: class repr
        :rtype: dictionnary
        """
        return {
            "Task": self.task.__content__(),
            "User": self.user.__content__(),
            "Assistant": self.assistant.__content__(),
        }

    @property
    def parameters(self):
        return {
            **self.task._parameters,
            **self.user._parameters,
            **self.assistant._parameters,
        }

    @property
    def turn_number(self):
        """turn_number

        The turn number in the game (0 to 3)

        :return: turn number
        :rtype: numpy.ndarray
        """
        return self.game_state["game_info"]["turn_index"]

    @turn_number.setter
    def turn_number(self, value):
        self._turn_number = value
        self.game_state["game_info"]["turn_index"] = value

    @property
    def round_number(self):
        """round_number

        The round number in the game (0 to N)

        :return: turn number
        :rtype: numpy.ndarray
        """
        return self.game_state["game_info"]["round_index"]

    @round_number.setter
    def round_number(self, value):
        self._round_number = value
        self.game_state["game_info"]["round_index"] = value

    @property
    def state(self):
        return self.game_state

    def reset(
        self,
        go_to=None,
        start_after=None,
        task=True,
        user=True,
        assistant=True,
        dic={},
        random_reset=False,
    ):
        """Reset bundle.

        1. Reset the game and start at a specific turn number.
        2. select which components to reset
        3. forced reset mechanism using dictionnaries


        Example:

        .. code-block:: python

            new_target_value = self.game_state["task_state"]["targets"]
            new_fixation_value = self.game_state["task_state"]["fixation"]
            )
            reset_dic = {"task_state": {"targets": new_target_value, "fixation": new_fixation_value}}
            self.reset(dic=reset_dic, turn = 1)

        Will set the substates "targets" and "fixation" of state "task_state" to some value.


        .. note ::

            If subclassing BaseBundle, make sure to call super().reset() in the new reset method.


        :param turn: game turn number. Can also be set globally at the bundle level by passing the "reset_turn" keyword argument, defaults to 0
        :type turn: int, optional
        :param start_after: which turn to start at (allows skipping some turns during reset), defaults to 0
        :type start_after: int, optional
        :param task: reset task?, defaults to True
        :type task: bool, optional
        :param user: reset user?, defaults to True
        :type user: bool, optional
        :param assistant: reset assistant?, defaults to True
        :type assistant: bool, optional
        :param dic: reset_dic, defaults to {}
        :type dic: dict, optional
        :param random_reset: whether during resetting values should be randomized or not if not set by a reset dic, default to False
        :type random_reset: bool, optional
        :return: new game state
        :rtype: :py:class:`State<coopihc.base.State.State>`
        """

        if go_to is None:
            go_to = self._reset_go_to

        if start_after is None:
            start_after = self._reset_start_after

        random_reset = self._reset_random or random_reset

        if task:
            task_dic = dic.get("task_state")
            self.task._base_reset(
                dic=task_dic,
                random=random_reset,
            )

        if user:
            user_dic = dic.get("user_state")
            self.user._base_reset(
                dic=user_dic,
                random=random_reset,
            )

        if assistant:
            assistant_dic = dic.get("assistant_state")
            self.assistant._base_reset(
                dic=assistant_dic,
                random=random_reset,
            )

        self.round_number = 0

        if not isinstance(go_to, (numpy.integer, int)):
            go_to = self.turn_dict[go_to]
        if not isinstance(start_after, (numpy.integer, int)):
            start_after = self.turn_dict[start_after]

        self.turn_number = go_to

        if go_to == 0 and start_after + 1 == 0:
            return self.game_state
        if start_after <= go_to:
            if go_to >= 1 and start_after + 1 <= 1:
                self._user_first_half_step()
            if go_to >= 2 and start_after + 1 <= 2:
                user_action, _ = self.user.take_action(increment_turn=False)
                self.user.action = user_action
                self._user_second_half_step(user_action)
            if go_to >= 3 and start_after + 1 <= 3:
                self._assistant_first_half_step()
        else:
            raise ValueError(
                f"start_after ({start_after}) can not be after go_to ({go_to}). You can likely use a combination of reset and step to achieve what you are looking for"
            )

        return self.game_state

    def quarter_step(self, user_action=None, assistant_action=None, **kwargs):
        return self.step(
            user_action=user_action,
            assistant_action=assistant_action,
            go_to=(int(self.turn_number) + 1) % 4,
        )

    def step(self, user_action=None, assistant_action=None, go_to=None, **kwargs):
        """Play a round

        Play a round of the game. A round consists in 4 turns. If go_to is not None, the round is only played until that turn.
        If a user action and assistant action are passed as arguments, then these are used as actions to play the round. Otherwise, these actions are sampled from each agent's policy.

        :param user action: user action
        :type: any
        :param assistant action: assistant action
        :type: any
        :param go_to: turn at which round stops, defaults to None
        :type go_to: int, optional
        :return: gamestate, reward, game finished flag
        :rtype: tuple(:py:class:`State<coopihc.base.State.State>`, collections.OrderedDict, boolean)
        """

        if go_to is None:
            go_to = int(self.turn_number)

        if not isinstance(go_to, (numpy.integer, int)):
            go_to = self.turn_dict[go_to]

        _started = False
        rewards = {}
        rewards["user_observation_reward"] = 0
        rewards["user_inference_reward"] = 0
        rewards["user_policy_reward"] = 0
        rewards["first_task_reward"] = 0
        rewards["assistant_observation_reward"] = 0
        rewards["assistant_inference_reward"] = 0
        rewards["assistant_policy_reward"] = 0
        rewards["second_task_reward"] = 0

        while self.turn_number != go_to or (not _started):

            _started = True
            # User observes and infers
            if self.turn_number == 0 and "no-user" != self.kwargs.get("name"):
                (
                    user_obs_reward,
                    user_infer_reward,
                ) = self._user_first_half_step()
                (
                    rewards["user_observation_reward"],
                    rewards["user_inference_reward"],
                ) = (user_obs_reward, user_infer_reward)

            # User takes action and receives reward from task
            elif self.turn_number == 1 and "no-user" != self.kwargs.get("name"):
                if user_action is None:
                    user_action, user_policy_reward = self.user.take_action(
                        increment_turn=False
                    )
                else:
                    self.user.action = user_action
                    user_policy_reward = 0

                task_reward, is_done = self._user_second_half_step(user_action)
                rewards["user_policy_reward"] = user_policy_reward
                rewards["first_task_reward"] = task_reward
                if is_done:
                    return self.game_state, rewards, is_done

            elif self.turn_number == 2 and "no-assistant" == self.kwargs.get("name"):
                self.round_number = self.round_number + 1

            # Assistant observes and infers
            elif self.turn_number == 2 and "no-assistant" != self.kwargs.get("name"):
                (
                    assistant_obs_reward,
                    assistant_infer_reward,
                ) = self._assistant_first_half_step()
                (
                    rewards["assistant_observation_reward"],
                    rewards["assistant_inference_reward"],
                ) = (assistant_obs_reward, assistant_infer_reward)

            # Assistant takes action and receives reward from task
            elif self.turn_number == 3 and "no-assistant" != self.kwargs.get("name"):
                if assistant_action is None:
                    (
                        assistant_action,
                        assistant_policy_reward,
                    ) = self.assistant.take_action(increment_turn=False)
                else:
                    self.assistant.action = assistant_action
                    assistant_policy_reward = 0

                task_reward, is_done = self._assistant_second_half_step(
                    assistant_action
                )
                rewards["assistant_policy_reward"] = assistant_policy_reward
                rewards["second_task_reward"] = task_reward
                if is_done:
                    return self.game_state, rewards, is_done

                self.round_number = self.round_number + 1

            self.turn_number = (self.turn_number + 1) % 4

        return self.game_state, rewards, False

    def render(self, mode, *args, **kwargs):
        """render

        Combines all render methods.

        :param mode: "text" or "plot"
        :param type: string

        :meta public:
        """

        self.rendered_mode = mode
        if "text" in mode:
            print("\n")
            print("Round number {}".format(self.round_number.tolist()))
            print("Task Render")
            self.task.render(mode="text", *args, **kwargs)
            print("User Render")
            self.user.render(mode="text", *args, **kwargs)
            print("Assistant Render")
            self.assistant.render(mode="text", *args, **kwargs)
        if "log" in mode:
            self.task.render(mode="log", *args, **kwargs)
            self.user.render(mode="log", *args, **kwargs)
            self.assistant.render(mode="log", *args, **kwargs)
        if "plot" in mode:
            if self.active_render_figure:
                plt.pause(self.playspeed)
                self.task.render(
                    ax_task=self.axtask,
                    ax_user=self.axuser,
                    ax_assistant=self.axassistant,
                    mode="plot",
                    **kwargs,
                )
                self.user.render(
                    ax_task=self.axtask,
                    ax_user=self.axuser,
                    ax_assistant=self.axassistant,
                    mode="plot",
                    **kwargs,
                )
                self.assistant.render(
                    ax_task=self.axtask,
                    ax_user=self.axuser,
                    ax_assistant=self.axassistant,
                    mode="plot",
                    **kwargs,
                )
                self.fig.canvas.draw()
            else:
                self.active_render_figure = True
                self.fig = plt.figure()
                self.axtask = self.fig.add_subplot(self.figure_layout[0])
                self.axtask.set_title("Task State")
                self.axuser = self.fig.add_subplot(self.figure_layout[1])
                self.axuser.set_title("User State")
                self.axassistant = self.fig.add_subplot(self.figure_layout[2])
                self.axassistant.set_title("Assistant State")
                self.task.render(
                    ax_task=self.axtask,
                    ax_user=self.axuser,
                    ax_assistant=self.axassistant,
                    mode="plot",
                    **kwargs,
                )
                self.user.render(
                    ax_task=self.axtask,
                    ax_user=self.axuser,
                    ax_assistant=self.axassistant,
                    mode="plot",
                    **kwargs,
                )
                self.assistant.render(
                    ax_task=self.axtask,
                    ax_user=self.axuser,
                    ax_assistant=self.axassistant,
                    mode="plot",
                    **kwargs,
                )
                self.fig.show()

            plt.tight_layout()

        if not ("plot" in mode or "text" in mode):
            self.task.render(None, mode=mode, *args, **kwargs)
            self.user.render(None, mode=mode, *args, **kwargs)
            self.assistant.render(None, mode=mode, *args, **kwargs)

    def close(self):
        """close

        Close the bundle once the game is finished.
        """

        if self.active_render_figure:
            plt.close(self.fig)
            # self.active_render_figure = None

    def _user_first_half_step(self):
        """_user_first_half_step

        Turn 1, where the user observes the game state and updates its state via inference.

        :return: user observation and inference reward
        :rtype: tuple(float, float)
        """

        if not self.kwargs.get("onreset_deterministic_first_half_step"):
            user_obs_reward, user_infer_reward = self.user._agent_step()

        else:
            # Store the probabilistic rules
            store = self.user.observation_engine.extraprobabilisticrules
            # Remove the probabilistic rules
            self.user.observation_engine.extraprobabilisticrules = {}
            # Generate an observation without generating an inference
            user_obs_reward, user_infer_reward = self.user._agent_step(infer=False)
            # Reposition the probabilistic rules, and reset mapping
            self.user.observation_engine.extraprobabilisticrules = store
            self.user.observation_engine.mapping = None

        self.kwargs["onreset_deterministic_first_half_step"] = False

        return user_obs_reward, user_infer_reward

    def _user_second_half_step(self, user_action):
        """_user_second_half_step

        Turn 2, where the operaror takes an action.

        :param user_action: user action
        :param type: Any

        :return: task reward, task done?
        :rtype: tuple(float, boolean)
        """

        # Play user's turn in the task
        task_state, task_reward, is_done = self.task.base_on_user_action(
            user_action=user_action
        )

        return task_reward, is_done

    def _assistant_first_half_step(self):
        """_assistant_first_half_step

        Turn 3, where the assistant observes the game state and updates its state via inference.

        :return: assistant observation and inference reward
        :rtype: tuple(float, float)
        """
        (
            assistant_obs_reward,
            assistant_infer_reward,
        ) = self.assistant._agent_step()

        return assistant_obs_reward, assistant_infer_reward

    def _assistant_second_half_step(self, assistant_action):
        """_assistant_second_half_step

        Turn 4, where the assistant takes an action.

        :param user_action: assistant action
        :param type: Any

        :return: task reward, task done?
        :rtype: tuple(float, boolean)
        """

        # Play assistant's turn in the task

        task_state, task_reward, is_done = self.task.base_on_assistant_action(
            assistant_action=assistant_action
        )

        return task_reward, is_done

    def _on_user_action(self, *args):
        """Turns 1 and 2

        :param \*args: either provide the user action or not. If no action is provided the action is determined by the agent's policy using sample()
        :param type: (None or list)
        :return: user observation, inference, policy and task rewards, game is done flag
        :return type: tuple(float, float, float, float, bool)
        """
        user_obs_reward, user_infer_reward = self._user_first_half_step()
        try:
            # If human input is provided
            user_action = args[0]
        except IndexError:
            # else sample from policy
            user_action, user_policy_reward = self.user.take_action(
                increment_turn=False
            )

        self.user.action = user_action

        task_reward, is_done = self._user_second_half_step(user_action)

        return (
            user_obs_reward,
            user_infer_reward,
            user_policy_reward,
            task_reward,
            is_done,
        )

    def _on_assistant_action(self, *args):
        """Turns 3 and 4

        :param \*args: either provide the assistant action or not. If no action is provided the action is determined by the agent's policy using sample()
        :param type: (None or list)
        :return: assistant observation, inference, policy and task rewards, game is done flag
        :return type: tuple(float, float, float, float, bool)
        """
        (
            assistant_obs_reward,
            assistant_infer_reward,
        ) = self._assistant_first_half_step()

        try:
            # If human input is provided
            assistant_action = args[0]
        except IndexError:
            # else sample from policy
            (
                assistant_action,
                assistant_policy_reward,
            ) = self.assistant.take_action(increment_turn=False)

        self.assistant.action = assistant_action

        task_reward, is_done = self._assistant_second_half_step(assistant_action)
        return (
            assistant_obs_reward,
            assistant_infer_reward,
            assistant_policy_reward,
            task_reward,
            is_done,
        )
