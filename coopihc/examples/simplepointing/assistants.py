import coopihc
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.policy.BIGDiscretePolicy import BIGDiscretePolicy
from coopihc.inference.GoalInferenceWithUserPolicyGiven import (
    GoalInferenceWithUserPolicyGiven,
)
from coopihc.base.Space import Space
from coopihc.base.StateElement import StateElement
from coopihc.base.elements import discrete_array_element, array_element, cat_element

import numpy
import copy


class ConstantCDGain(BaseAgent):
    """A Constant CD Gain transfer function.

    Here the assistant just picks a fixed modulation.

    :param gain: (float) constant CD gain.

    :meta public:
    """

    def __init__(self, gain):
        self.gain = gain

        super().__init__("assistant")

    def finit(self):

        self.policy.action_state["action"] = array_element(
            init=self.gain,
            low=numpy.full((1, self.bundle.task.dim), self.gain),
            high=numpy.full((1, self.bundle.task.dim), self.gain),
        )


class BIGGain(BaseAgent):
    def __init__(self):

        super().__init__(
            "assistant", agent_inference_engine=GoalInferenceWithUserPolicyGiven()  #
        )

    def finit(self):
        action_state = self.bundle.game_state["assistant_action"]
        action_state["action"] = discrete_array_element(
            init=0, low=0, high=self.bundle.task.gridsize, out_of_bounds_mode="error"
        )

        user_policy_model = copy.deepcopy(self.bundle.user.policy)
        agent_policy = BIGDiscretePolicy(action_state, user_policy_model)
        self.attach_policy(agent_policy)
        self.inference_engine.attach_policy(user_policy_model)

        self.state["beliefs"] = array_element(
            init=1 / self.bundle.task.number_of_targets,
            low=numpy.zeros((self.bundle.task.number_of_targets,)),
            high=numpy.ones((self.bundle.task.number_of_targets,)),
            out_of_bounds_more="error",
        )

    def reset(self, dic=None):
        self.state["beliefs"][...] = numpy.array(
            [
                1 / self.bundle.task.number_of_targets
                for i in range(self.bundle.task.number_of_targets)
            ]
        )

        # change theta for inference engine
        set_theta = [
            {
                ("user_state", "goal"): discrete_array_element(
                    init=t, low=0, high=self.bundle.task.gridsize
                )
            }
            for t in self.bundle.task.state["targets"]
        ]

        self.inference_engine.attach_set_theta(set_theta)
        self.policy.attach_set_theta(set_theta)

        def transition_function(assistant_action, observation):
            """What future observation will the user see due to assistant action"""
            # always do this
            observation["assistant_action"]["action"] = assistant_action
            # specific to BIGpointer
            observation["task_state"]["position"] = assistant_action

            return observation

        self.policy.attach_transition_function(transition_function)

    def render(self, *args, **kwargs):
        mode = kwargs.get("mode")
        if mode is None:
            mode = "text"
        try:
            axtask, axuser, axassistant = args
            self.inference_engine.render(axassistant, mode=mode)
        except ValueError:
            self.inference_engine.render(mode=mode)
