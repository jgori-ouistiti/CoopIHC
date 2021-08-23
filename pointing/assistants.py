import core
from core.agents import BaseAgent
from core.policy import BasePolicy,  BIGDiscretePolicy
from core.inference import GoalInferenceWithOperatorPolicyGiven
from core.space import State, StateElement

import gym


class ConstantCDGain(BaseAgent):
    """ A Constant CD Gain transfer function.

    Here the assistant just picks a fixed modulation.

    :param gain: (float) constant CD gain.

    :meta public:
    """
    def __init__(self, gain):
        self.gain = gain


        super().__init__( 'assistant',
                                observation_engine = None,
                                inference_engine = None#,
                                # policy = agent_policy
                                )

    def finit(self):
        action_space = [core.space.Box(low = self.gain, high = self.gain, shape = (self.bundle.task.dim,))]
        self.policy.action_state['action']['spaces'] = action_space
        self.policy.action_state['action']['clipping_mode'] = 'clip'

class BIGGain(BaseAgent):
    def __init__(self):

        super().__init__(       'assistant',
                                inference_engine = GoalInferenceWithOperatorPolicyGiven() #
                                )


    def finit(self):

        assistant_action_space = [core.space.Discrete(self.bundle.task.gridsize)]
        operator_policy_model = self.bundle.operator.policy

        action_state = self.bundle.game_state['assistant_action']

        agent_policy = BIGDiscretePolicy(       action_state,
                                                operator_policy_model,
                                                assistant_action_space,
                                                )

        self.attach_policy(agent_policy)
        self.inference_engine.attach_policy(agent_policy.operator_policy_model)




    def reset(self, dic = None):
        if dic is None:
            super().reset()

        self.state['Beliefs'] = StateElement(values = [1/self.bundle.task.number_of_targets for i in range(self.bundle.task.number_of_targets)], spaces = [core.space.Box(0, 1, shape = (1,)) for i in range(self.bundle.task.number_of_targets)], possible_values = None)

        # change theta for inference engine

        set_theta = [{('operator_state', 'goal'): StateElement(values = [t],
                spaces = [core.space.Discrete(self.bundle.task.gridsize)],
                possible_values =  [list(range(self.bundle.task.gridsize))] )  } for t in self.bundle.task.state['targets']['values'] ]

        self.inference_engine.attach_set_theta(set_theta)
        self.policy.attach_set_theta(set_theta)

        if dic is not None:
            super().reset(dic = dic)

        def transition_function(assistant_action, observation):
            """ What future observation will the user see due to assistant action
            """
            # always do this
            observation['assistant_action']['action'] = assistant_action
            # specific to BIGpointer
            observation['task_state']['position'] = assistant_action

            return observation

        self.policy.attach_transition_function(transition_function)


    def render(self, *args, **kwargs):
        mode = kwargs.get('mode')
        if mode is None:
            mode = 'text'
        try:
            axtask, axoperator, axassistant = args
            self.inference_engine.render(axassistant, mode = mode)
        except ValueError:
            self.inference_engine.render(mode = mode)
