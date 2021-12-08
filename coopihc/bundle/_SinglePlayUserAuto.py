from coopihc.bundle._Bundle import _Bundle


class SinglePlayUserAuto(_Bundle):
    """Same as SinglePlayUser, but this time the user action is obtained by sampling the user policy.

    :param task: (coopihc.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (coopihc.agents.BaseAgent) an user, which is a subclass of BaseAgent
    :param kwargs: additional controls to account for some specific subcases. See Doc for a full list.

    :meta public:
    """

    def __init__(self, task, user, **kwargs):
        super().__init__(task=task, user=user, **kwargs)
        self.action_space = None
        self.kwargs = kwargs

    @property
    def observation(self):
        return self.user.observation

    def reset(self, dic={}, **kwargs):
        """Reset the bundle. A first observation and inference is performed.

        :param args: see Bundle

        :meta public:
        """
        super().reset(dic=dic, **kwargs)

        if self.kwargs.get("start_at_action"):
            self._user_first_half_step()
            return self.observation

        return self.game_state
        # Return observation

    def step(self):
        """Play a step, user actions are obtained by sampling the agent's policy.

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        if not self.kwargs.get("start_at_action"):
            user_obs_reward, user_infer_reward = self._user_first_half_step()
        user_action, user_policy_reward = self.user._take_action()
        self.broadcast_action("user", user_action)

        first_task_reward, is_done = self._user_second_half_step(user_action)
        if is_done:
            return (
                self.observation,
                first_task_reward,
                is_done,
                [first_task_reward],
            )
        _, _, _, _ = self.task.base_assistant_step([0])
        if self.kwargs.get("start_at_action"):
            user_obs_reward, user_infer_reward = self._user_first_half_step()
        return (
            self.observation,
            sum([user_obs_reward, user_infer_reward, first_task_reward]),
            is_done,
            [user_obs_reward, user_infer_reward, first_task_reward],
        )
