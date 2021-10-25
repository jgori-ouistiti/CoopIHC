from core.space import State, StateElement, Space
import numpy
import yaml
from collections import OrderedDict
import matplotlib.pyplot as plt


class _Bundle:

    """A bundle combines a task with an user and an assistant. All bundles are obtained by subclassing this main _Bundle class.

    A bundle will create the ``game_state`` by combining three states of the task, the user and the assistant as well as the turn index. It also takes care of adding the assistant action substate to the user state and vice-versa.
    It also takes care of rendering each of the three component in a single place.

    Bundle subclasses should only have to redefine the step() and reset() methods.


    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (core.agents.BaseAgent) an user, which is a subclass of BaseAgent
    :param assistant: (core.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """

    def __init__(self, task, user, assistant, **kwargs):
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
            values=[0],
            spaces=Space([numpy.array([0, 1, 2, 3])], dtype=numpy.int8),
        )

        self.game_state["turn_index"] = turn_index
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

        self.round_number = 0

        # Needed for render
        self.active_render_figure = None
        self.figure_layout = [211, 223, 224]
        self.rendered_mode = None
        self.render_perm = False
        self.playspeed = 0.1

    def __repr__(self):
        return "{}\n".format(self.__class__.__name__) + yaml.safe_dump(
            self.__content__()
        )

    def __content__(self):
        return {
            "Task": self.task.__content__(),
            "User": self.user.__content__(),
            "Assistant": self.assistant.__content__(),
        }

    @property
    def turn_number(self):
        return self.game_state["turn_index"]["values"][0]

    @turn_number.setter
    def turn_number(self, value):
        self._turn_number = value
        self.game_state["turn_index"]["values"] = numpy.array(value)

    def reset(self, turn=0, task=True, user=True, assistant=True, dic={}):
        """Reset the bundle.

        When subclassing Bundle, make sure to call super().reset() in the new reset method.

        :param dic: (dictionnary) Reset the bundle with a game_state

        :return: (list) Flattened game_state

        :meta private:
        """
        if task:
            task_dic = dic.get("task_state")
            task_state = self.task._base_reset(dic=task_dic)

        if user:
            user_dic = dic.get("user_state")
            user_state = self.user._base_reset(dic=user_dic)

        if assistant:
            assistant_dic = dic.get("assistant_state")
            assistant_state = self.assistant._base_reset(dic=assistant_dic)

        self.turn_number = turn
        if turn == 0:
            return self.game_state
        if turn >= 1:
            self._user_first_half_step()
        if turn >= 2:
            user_action, _ = self.user.take_action()
            self.broadcast_action("user", user_action)
            self._user_second_half_step(user_action)
        if turn >= 3:
            self._assistant_first_half_step()

        return self.game_state

    def step(self, *args, go_to_turn=None, **kwargs):
        # step() was called
        if not args:
            user_action, assistant_action = None, None
        elif len(args) == 1:
            if self.kwargs.get("name") == "no-assistant":
                user_action, assistant_action = args[0], None
            elif self.kwargs.get("name") == "no-user":
                user_action, assistant_action = None, args[0]
            else:
                raise AttributeError(
                    "Passing a single action is only allowed when the game is played with a single agent."
                )
        # step(user_action, None) or step(None, assistant_action) or step(user_action, assistant_action) was called
        else:
            user_action, assistant_action = args

        if go_to_turn is None:
            go_to_turn = self.turn_number

        _started = False
        rewards = OrderedDict()
        rewards["user_observation_reward"] = 0
        rewards["user_inference_reward"] = 0
        rewards["first_task_reward"] = 0
        rewards["assistant_observation_reward"] = 0
        rewards["assistant_inference_reward"] = 0
        rewards["second_task_reward"] = 0

        while self.turn_number != go_to_turn or (not _started):
            _started = True
            if self.turn_number == 0:
                (
                    user_obs_reward,
                    user_infer_reward,
                ) = self._user_first_half_step()
                (
                    rewards["user_observation_reward"],
                    rewards["user_inference_reward"],
                ) = (user_obs_reward, user_infer_reward)

            elif self.turn_number == 1:
                if user_action is None:
                    user_action, user_policy_reward = self.user._take_action()
                else:
                    user_policy_reward = 0
                self.broadcast_action("user", user_action)
                task_reward, is_done = self._user_second_half_step(user_action)
                rewards["first_task_reward"] = task_reward
                if is_done:
                    return self.game_state, rewards, is_done
            elif self.turn_number == 2:
                (
                    assistant_obs_reward,
                    assistant_infer_reward,
                ) = self._assistant_first_half_step()
                (
                    rewards["assistant_observation_reward"],
                    rewards["assistant_inference_reward"],
                ) = (assistant_obs_reward, assistant_infer_reward)
            elif self.turn_number == 3:
                if assistant_action is None:
                    (
                        assistant_action,
                        assistant_policy_reward,
                    ) = self.assistant._take_action()
                else:
                    assistant_policy_reward = 0
                self.broadcast_action("assistant", assistant_action)
                task_reward, is_done = self._assistant_second_half_step(
                    assistant_action
                )
                rewards["second_task_reward"] = task_reward
                if is_done:
                    return self.game_state, rewards, is_done

            self.turn_number = (self.turn_number + 1) % 4

        self.round_number += 1
        return self.game_state, rewards, False

    def render(self, mode, *args, **kwargs):
        """Combines all render methods.

        :param mode: (str) text or plot

        :meta public:
        """
        self.rendered_mode = mode
        if "text" in mode:
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
                    mode=mode,
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
        """Close bundle. Call this after the bundle returns is_done True.

        :meta public:
        """
        if self.active_render_figure:
            plt.close(self.fig)
            self.active_render_figure = None

    def _user_first_half_step(self):
        """This is the first half of the user step, where the operaror observes the game state and updates its state via inference.

        :return: user_obs_reward, user_infer_reward (float, float): rewards for the observation and inference process.

        :meta public:
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
        """This is the second half of the user step. The operaror takes an action, which is applied to the task leading to a new game state.

        :param user_action: (list) user action

        :return: task_reward, is_done (float, bool): rewards returned by the task and boolean that determines whether the task is finished.

        :meta public:
        """

        # Play user's turn in the task
        task_state, task_reward, is_done, _ = self.task.base_user_step(user_action)

        # update task state (likely not needed, remove ?)
        self.broadcast_state("user", "task_state", task_state)

        return task_reward, is_done

    def _assistant_first_half_step(self):
        """This is the first half of the assistant step, where the assistant observes the game state and updates its state via inference.

        :return: assistant_obs_reward, assistant_infer_reward (float, float): rewards for the observation and inference process.

        :meta public:
        """

        (
            assistant_obs_reward,
            assistant_infer_reward,
        ) = self.assistant._agent_step()

        return assistant_obs_reward, assistant_infer_reward

    def _assistant_second_half_step(self, assistant_action):
        """This is the second half of the assistant step. The assistant takes an action, which is applied to the task leading to a new game state.

        :param assistant_action: (list) assistant action

        :return: task_reward, is_done (float, bool): rewards returned by the task and boolean that determines whether the task is finished.

        :meta public:
        """
        # update action_state

        # Play assistant's turn in the task

        task_state, task_reward, is_done, _ = self.task.base_assistant_step(
            assistant_action
        )
        # update task state
        self.broadcast_state("assistant", "task_state", task_state)

        return task_reward, is_done

    def _user_step(self, *args):
        """Combines the first and second half step of the user.

        :param args: (None or list) either provide the user action or not. If no action is provided the action is determined by the agent's policy using sample()

        :return: user_obs_reward, user_infer_reward, task_reward, is_done (float, float, float, bool) The returns for the two half steps combined.

        :meta public:
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
        """Combines the first and second half step of the assistant.

        :param args: (None or list) either provide the assistant action or not. If no action is provided the action is determined by the agent's policy using sample()

        :return: assistant_obs_reward, assistant_infer_reward, task_reward, is_done (float, float, float, bool) The returns for the two half steps combined.

        :meta public:
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
        self.game_state[state_key] = state
        getattr(self, role).observation[state_key] = state

    def broadcast_action(self, role, action):
        # update game state and observations
        if isinstance(action, StateElement):
            getattr(self, role).policy.action_state["action"] = action
            getattr(self, role).observation["{}_action".format(role)]["action"] = action
        else:
            getattr(self, role).policy.action_state["action"]["values"] = action
            getattr(self, role).observation["{}_action".format(role)]["action"][
                "values"
            ] = action
