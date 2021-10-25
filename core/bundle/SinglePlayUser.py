from core.bundle import _Bundle


class SinglePlayUser(_Bundle):
    """A bundle without assistant. This is used e.g. to model psychophysical tasks such as perception, where there is no real interaction loop with a computing device.

    :param task: (core.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (core.agents.BaseAgent) an user, which is a subclass of BaseAgent

    :meta public:
    """

    def __init__(self, task, user, **kwargs):
        super().__init__(task=task, user=user, **kwargs)

    @property
    def observation(self):
        return self.user.observation

    def reset(self, dic={}, **kwargs):
        """Reset the bundle. A first observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        full_obs = super().reset(dic=dic, **kwargs)
        self._user_first_half_step()
        return self.observation

    def step(self, user_action):
        """Play a step, user actions are given externally in the step() method.

        :param user_action: (list) user action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """

        super().step(user_action)

        self.broadcast_action("user", user_action)
        first_task_reward, is_done = self._user_second_half_step(user_action)
        if is_done:
            return (
                self.user.inference_engine.buffer[-1],
                first_task_reward,
                is_done,
                [first_task_reward],
            )
        self.task.base_assistant_step([None])
        user_obs_reward, user_infer_reward = self._user_first_half_step()
        return (
            self.user.inference_engine.buffer[-1],
            sum([user_obs_reward, user_infer_reward, first_task_reward]),
            is_done,
            [user_obs_reward, user_infer_reward, first_task_reward],
        )
