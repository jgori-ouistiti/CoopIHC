from coopihc.space.Space import Space
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement

import numpy
import yaml
from collections import OrderedDict
import matplotlib.pyplot as plt
import copy


class _Bundle:
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

    def __init__(self, task, user, assistant, *args, **kwargs):
        self.kwargs = kwargs
        self.task = task
        self.task.bundle = self
        self.user = user
        self.user.bundle = self
        self.assistant = assistant
        self.assistant.bundle = self

        # Form complete game state
        self.game_state = State()

        turn_index = StateElement(
            numpy.array([0]),
            Space(numpy.array([0, 1, 2, 3], dtype=numpy.int8), "discrete"),
            out_of_bounds_mode="raw",
        )
        round_index = StateElement(
            numpy.array([0]),
            Space(numpy.array([0, 1], dtype=numpy.int8), "discrete"),
            out_of_bounds_mode="raw",
        )

        self.game_state["game_info"] = State()
        self.game_state["game_info"]["turn_index"] = turn_index
        self.game_state["game_info"]["round_index"] = round_index
        self.game_state["task_state"] = task.state
        self.game_state["user_state"] = user.state
        self.game_state["assistant_state"] = assistant.state

        if user.policy is not None:
            self.game_state["user_action"] = user.policy.action_state
        else:
            self.game_state["user_action"] = State()
            self.game_state["user_action"]["action"] = StateElement()
        if assistant.policy is not None:
            self.game_state["assistant_action"] = assistant.policy.action_state
        else:
            self.game_state["assistant_action"] = State()
            self.game_state["assistant_action"]["action"] = StateElement()

        self.task.finit()
        self.user.finit()
        self.assistant.finit()

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
        self.game_state["game_info"]["turn_index"][:] = value

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
        self.game_state["game_info"]["round_index"][:] = value

    def reset(self, turn=0, task=True, user=True, assistant=True, dic={}):
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

            If subclassing _Bundle, make sure to call super().reset() in the new reset method.


        :param turn: game turn number, defaults to 0
        :type turn: int, optional
        :param task: reset task?, defaults to True
        :type task: bool, optional
        :param user: reset user?, defaults to True
        :type user: bool, optional
        :param assistant: reset assistant?, defaults to True
        :type assistant: bool, optional
        :param dic: reset_dic, defaults to {}
        :type dic: dict, optional
        :return: new game state
        :rtype: :py:class:`State<coopihc.space.State.State>`
        """

        if task:
            task_dic = dic.get("task_state")
            self.task._base_reset(dic=task_dic)

        if user:
            user_dic = dic.get("user_state")
            self.user._base_reset(dic=user_dic)

        if assistant:
            assistant_dic = dic.get("assistant_state")
            self.assistant._base_reset(dic=assistant_dic)

        self.round_number[:] = 0

        self.turn_number[:] = turn

        if turn == 0:
            return self.game_state
        if turn >= 1:
            self._user_first_half_step()
        if turn >= 2:
            user_action, _ = self.user._take_action()
            self.broadcast_action("user", user_action)
            self._user_second_half_step(user_action)
        if turn >= 3:
            self._assistant_first_half_step()

        return self.game_state

    def step(self, user_action=None, assistant_action=None, go_to_turn=None, **kwargs):
        """Play a round

        Play a round of the game. A round consists in 4 turns. If go_to_turn is not None, the round is only played until that turn.
        If a user action and assistant action are passed as arguments, then these are used as actions to play the round. Otherwise, these actions are sampled from each agent's policy.

        :param user action: user action
        :type: any
        :param assistant action: assistant action
        :type: any
        :param go_to_turn: turn at which round stops, defaults to None
        :type go_to_turn: int, optional
        :return: gamestate, reward, game finished flag
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, collections.OrderedDict, boolean)
        """

        if go_to_turn is None:
            go_to_turn = self.turn_number.squeeze().tolist()

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

        while self.turn_number != go_to_turn or (not _started):

            _started = True
            # User observes and infers
            if self.turn_number == 0 and "no-user" not in self.kwargs.get("name"):
                (
                    user_obs_reward,
                    user_infer_reward,
                ) = self._user_first_half_step()
                (
                    rewards["user_observation_reward"],
                    rewards["user_inference_reward"],
                ) = (user_obs_reward, user_infer_reward)

            # User takes action and receives reward from task
            elif self.turn_number == 1 and "no-user" not in self.kwargs.get("name"):
                if user_action is None:
                    user_action, user_policy_reward = self.user._take_action()
                else:
                    # Convert action to stateElement
                    if not isinstance(user_action, StateElement):
                        se_action = copy.copy(self.user.action)
                        se_action[:] = user_action
                        user_action = se_action
                    user_policy_reward = 0
                self.broadcast_action("user", user_action)
                task_reward, is_done = self._user_second_half_step(user_action)
                rewards["user_policy_reward"] = user_policy_reward
                rewards["first_task_reward"] = task_reward
                if is_done:
                    return self.game_state, rewards, is_done

            elif self.turn_number == 2 and "no-assistant" in self.kwargs.get("name"):
                self.round_number = self.round_number + 1

            # Assistant observes and infers
            elif self.turn_number == 2 and "no-assistant" not in self.kwargs.get(
                "name"
            ):
                (
                    assistant_obs_reward,
                    assistant_infer_reward,
                ) = self._assistant_first_half_step()
                (
                    rewards["assistant_observation_reward"],
                    rewards["assistant_inference_reward"],
                ) = (assistant_obs_reward, assistant_infer_reward)

            # Assistant takes action and receives reward from task
            elif self.turn_number == 3 and "no-assistant" not in self.kwargs.get(
                "name"
            ):
                if assistant_action is None:
                    (
                        assistant_action,
                        assistant_policy_reward,
                    ) = self.assistant._take_action()
                else:
                    # Convert action to stateElement
                    if not isinstance(assistant_action, StateElement):
                        se_action = copy.copy(self.assistant.action)
                        se_action[:] = assistant_action
                        assistant_action = se_action
                    assistant_policy_reward = 0
                self.broadcast_action("assistant", assistant_action)
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
            print("Round number {}".format(self.round_number.squeeze().tolist()))
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
                    self.axtask,
                    self.axuser,
                    self.axassistant,
                    mode="plot",
                    *args,
                    **kwargs,
                )
                self.user.render(
                    self.axtask,
                    self.axuser,
                    self.axassistant,
                    mode="plot",
                    *args,
                    **kwargs,
                )
                self.assistant.render(
                    self.axtask,
                    self.axuser,
                    self.axassistant,
                    mode="plot",
                    *args,
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
                    self.axtask,
                    self.axuser,
                    self.axassistant,
                    mode="plot",
                    *args,
                    **kwargs,
                )
                self.user.render(
                    self.axtask,
                    self.axuser,
                    self.axassistant,
                    *args,
                    mode="plot",
                    **kwargs,
                )
                self.assistant.render(
                    self.axtask,
                    self.axuser,
                    self.axassistant,
                    *args,
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
        task_state, task_reward, is_done = self.task.base_user_step(user_action)

        # update task state (likely not needed, remove ?)
        self.broadcast_state("user", "task_state", task_state)

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
        # update action_state

        # Play assistant's turn in the task

        task_state, task_reward, is_done = self.task.base_assistant_step(
            assistant_action
        )
        # update task state
        self.broadcast_state("assistant", "task_state", task_state)

        return task_reward, is_done

    def _user_step(self, *args):
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
            user_action, user_policy_reward = self.user.take_action()

        self.broadcast_action("user", user_action)

        task_reward, is_done = self._user_second_half_step(user_action)

        return (
            user_obs_reward,
            user_infer_reward,
            user_policy_reward,
            task_reward,
            is_done,
        )

    def _assistant_step(self, *args):
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
            ) = self.assistant.take_action()

        self.broadcast_action("assistant", assistant_action)

        task_reward, is_done = self._assistant_second_half_step(assistant_action)
        return (
            assistant_obs_reward,
            assistant_infer_reward,
            assistant_policy_reward,
            task_reward,
            is_done,
        )

    def broadcast_state(self, role, state_key, state):
        """broadcast state

        Broadcast a state value to the gamestate and update the agent's observation.

        :param role: "user" or "assistant"
        :type role: string
        :param state_key: state key in gamestate
        :type state_key: string
        :param state: new state value
        :type state: :py:class:`State<coopihc.space.State.State>`
        """
        self.game_state[state_key] = state
        getattr(self, role).observation[state_key] = state

    def broadcast_action(self, role, action):
        """broadcast action

        Broadcast an action to the gamestate and update the agent's policy.

        :param role: "user" or "assistant"
        :type role: string
        :param action: action
        :type action: Any
        """

        getattr(self, role).policy.action_state["action"] = action
        try:
            getattr(self, role).observation["{}_action".format(role)]["action"] = action
        except AttributeError:
            pass
