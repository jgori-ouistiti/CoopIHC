from coopihc.bundle._Bundle import _Bundle


class PlayNone(_Bundle):
    """A bundle which samples actions directly from users and assistants. It is used to evaluate an user and an assistant where the policies are already implemented.

    :param task: (coopihc.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (coopihc.agents.BaseAgent) an user, which is a subclass of BaseAgent
    :param assistant: (coopihc.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """

    def __init__(self, task, user, assistant, **kwargs):
        super().__init__(task, user, assistant, **kwargs)
        self.action_state = None

    # def reset(self, dic = {}, **kwargs):
    #     """ Reset the bundle.
    #
    #     :param args: see Bundle
    #
    #     :meta public:
    #     """
    #     full_obs = super().reset(dic = dic, **kwargs)

    def step(self):
        """Play a step, actions are obtained by sampling the agent's policies.

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        super().step(None)

        (
            user_obs_reward,
            user_infer_reward,
            user_policy_reward,
            first_task_reward,
            is_done,
        ) = self._user_step()
        if is_done:
            return (
                self.game_state,
                sum(
                    [
                        user_obs_reward,
                        user_infer_reward,
                        user_policy_reward,
                        first_task_reward,
                    ]
                ),
                is_done,
                [
                    user_obs_reward,
                    user_infer_reward,
                    user_policy_reward,
                    first_task_reward,
                ],
            )
        (
            assistant_obs_reward,
            assistant_infer_reward,
            assistant_policy_reward,
            second_task_reward,
            is_done,
        ) = self._assistant_step()
        return (
            self.game_state,
            sum(
                [
                    user_obs_reward,
                    user_infer_reward,
                    user_policy_reward,
                    first_task_reward,
                    assistant_obs_reward,
                    assistant_infer_reward,
                    assistant_policy_reward,
                    second_task_reward,
                ]
            ),
            is_done,
            [
                user_obs_reward,
                user_infer_reward,
                user_policy_reward,
                first_task_reward,
                assistant_obs_reward,
                assistant_infer_reward,
                assistant_policy_reward,
                second_task_reward,
            ],
        )
