from coopihc.bundle._Bundle import _Bundle


class PlayAssistant(_Bundle):
    """A bundle which samples oeprator actions directly from the user but uses assistant actions provided externally in the step() method.

    :param task: (coopihc.interactiontask.InteractionTask) A task, which is a subclass of InteractionTask
    :param user: (coopihc.agents.BaseAgent) an user, which is a subclass of BaseAgent
    :param assistant: (coopihc.agents.BaseAgent) an assistant, which is a subclass of BaseAgent

    :meta public:
    """

    def __init__(self, task, user, assistant, **kwargs):
        super().__init__(task, user, assistant, **kwargs)
        self.action_space = self.assistant.policy.action_state["action"]["spaces"]

        # assistant.policy.action_state['action'] = StateElement(
        #     values = None,
        #     spaces = [gym.spaces.Box(low = -numpy.inf, high = numpy.inf, shape = (1,)) for i in range(len(assistant.policy.action_state['action']))],
        #     possible_values = None
        #      )

    # def reset(self, dic = {}, **kwargs):
    #     """ Reset the bundle. A first  user step and assistant observation and inference is performed.
    #
    #     :param args: see Bundle
    #
    #     :meta public:
    #     """
    #     full_obs = super().reset(dic = dic, **kwargs)
    #     self._user_step()
    #     self._assistant_first_half_step()
    #     return self.assistant.inference_engine.buffer[-1]

    def step(self, assistant_action):
        """Play a step, user actions are obtained by sampling the agent's policy and assistant actions are given externally in the step() method.

        :param assistant_action: (list) assistant action

        :return: sum_rewards (float), is_done (bool), rewards (list). Returns the sum of all intermediate rewards, the is_done flag to indicate whether or not the task has finisged, and the list of intermediate rewards.

        :meta public:
        """
        super().step(assistant_action)

        self.broadcast_action("assistant", assistant_action, key="values")
        second_task_reward, is_done = self._assistant_second_half_step(assistant_action)
        if is_done:
            return (
                self.assistant.inference_engine.buffer[-1],
                second_task_reward,
                is_done,
                [second_task_reward],
            )
        (
            user_obs_reward,
            user_infer_reward,
            user_policy_reward,
            first_task_reward,
            is_done,
        ) = self._user_step()
        (
            assistant_obs_reward,
            assistant_infer_reward,
        ) = self._assistant_first_half_step()
        return (
            self.assistant.inference_engine.buffer[-1],
            sum(
                [
                    user_obs_reward,
                    user_infer_reward,
                    user_policy_reward,
                    first_task_reward,
                    assistant_obs_reward,
                    assistant_infer_reward,
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
                second_task_reward,
            ],
        )
