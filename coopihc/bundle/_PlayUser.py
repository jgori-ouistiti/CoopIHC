from coopihc.bundle._Bundle import _Bundle
import copy


class PlayUser(_Bundle):
    """A bundle which samples assistant actions directly from the assistant but uses user actions provided externally in the step() method.

    :param task: (coopihc.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (coopihc.agents.BaseAgent) an user, which is a subclass of BaseAgent
    :param assistant: (coopihc.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """

    def __init__(self, task, user, assistant, **kwargs):
        super().__init__(task, user, assistant, **kwargs)
        self.action_space = copy.copy(self.user.policy.action_state["action"]["spaces"])

    # def reset(self, dic = {}, **kwargs):
    #     """ Reset the bundle. A first observation and inference is performed.
    #
    #     :param args: see Bundle
    #
    #     :meta public:
    #     """
    #     full_obs = super().reset(dic = dic, **kwargs)
    #     self._user_first_half_step()
    #     return self.user.observation
    #     # return self.user.inference_engine.buffer[-1]

    def step(self, user_action):
        """Play a step, assistant actions are obtained by sampling the agent's policy and user actions are given externally in the step() method.

        :param user_action: (list) user action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        super().step(user_action)

        self.broadcast_action("user", user_action, key="values")

        first_task_reward, is_done = self._user_second_half_step(user_action)
        if is_done:
            return (
                self.user.inference_engine.buffer[-1],
                first_task_reward,
                is_done,
                [first_task_reward],
            )
        (
            assistant_obs_reward,
            assistant_infer_reward,
            assistant_policy_reward,
            second_task_reward,
            is_done,
        ) = self._assistant_step()
        user_obs_reward, user_infer_reward = self._user_first_half_step()
        return (
            self.user.inference_engine.buffer[-1],
            sum(
                [
                    user_obs_reward,
                    user_infer_reward,
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
                first_task_reward,
                assistant_obs_reward,
                assistant_infer_reward,
                assistant_policy_reward,
                second_task_reward,
            ],
        )
