from coopihc.bundle import _Bundle
from coopihc.space import StateElement


class PlayBoth(_Bundle):
    """A bundle which samples both actions directly from the user and assistant.

    :param task: (coopihc.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (coopihc.agents.BaseAgent) an user, which is a subclass of BaseAgent
    :param assistant: (coopihc.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """

    def __init__(self, task, user, assistant, **kwargs):
        super().__init__(task, user, assistant, **kwargs)

    # def reset(self, dic = {}, **kwargs):
    #     """ Reset the bundle. User observation and inference is performed.
    #
    #     :param args: see Bundle
    #
    #     :meta public:
    #     """
    #     full_obs = super().reset(dic = dic, **kwargs)
    #     self._user_first_half_step()
    #     return self.task.state

    def step(self, joint_action):
        """Play a step, user and assistant actions are given externally in the step() method.

        :param joint_action: (list) joint user assistant action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """

        super().step(joint_action)

        user_action, assistant_action = joint_action
        if isinstance(user_action, StateElement):
            self.broadcast_action("user", user_action)
        else:
            self.broadcast_action("user", user_action, key="values")

        first_task_reward, first_is_done = self._user_second_half_step(user_action)
        if first_is_done:
            return (
                self.task.state,
                first_task_reward,
                first_is_done,
                [first_task_reward],
            )

        (
            assistant_obs_reward,
            assistant_infer_reward,
        ) = self._assistant_first_half_step()

        if isinstance(assistant_action, StateElement):
            self.broadcast_action("assistant", assistant_action)
        else:
            self.broadcast_action("assistant", assistant_action, key="values")

        second_task_reward, second_is_done = self._assistant_second_half_step(
            assistant_action
        )

        user_obs_reward, user_infer_reward = self._user_first_half_step()
        return (
            self.task.state,
            sum(
                [
                    user_obs_reward,
                    user_infer_reward,
                    first_task_reward,
                    assistant_obs_reward,
                    assistant_infer_reward,
                    second_task_reward,
                ]
            ),
            second_is_done,
            [
                user_obs_reward,
                user_infer_reward,
                first_task_reward,
                assistant_obs_reward,
                assistant_infer_reward,
                second_task_reward,
            ],
        )
