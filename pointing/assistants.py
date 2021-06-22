from core.agents import BaseAgent
from core.policy import Policy,  BIGDiscretePolicy
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
        action_space = [gym.spaces.Discrete(1)]
        action_set = [[gain]]
        agent_policy = Policy(action_space = action_space, action_set = action_set)


        super().__init__( 'assistant',
                                policy = agent_policy,
                                observation_engine = None,
                                inference_engine = None
                                )

class BIGGain(BaseAgent):
    def __init__(self):

        super().__init__(       'assistant',
                                inference_engine = GoalInferenceWithOperatorPolicyGiven() #
                                )


    def finit(self):

        assistant_action_space = [gym.spaces.Discrete(self.bundle.task.gridsize)]
        operator_policy_model = self.bundle.operator.policy


        action_state = self.bundle.game_state['assistant_action']
        agent_policy = BIGDiscretePolicy(       action_state,
                                                assistant_action_space,
                                                operator_policy_model                                                )

        self.attach_policy(agent_policy)
        self.inference_engine.attach_policy(agent_policy.operator_policy_model)




    def reset(self, dic = None):
        if dic is None:
            super().reset()

        self.state['Beliefs'] = StateElement(values = [1/self.bundle.task.number_of_targets for i in range(self.bundle.task.number_of_targets)], spaces = [gym.spaces.Box(0, 1, shape = (1,)) for i in range(self.bundle.task.number_of_targets)], possible_values = None)

        # change theta for inference engine
        set_theta = [{('operator_state', 'Goal'): StateElement(values = [t],
                spaces = [gym.spaces.Discrete(self.bundle.task.gridsize)],
                possible_values =  self.bundle.task.state['Targets']['values'])  } for t in self.bundle.task.state['Targets']['values'] ]

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
            observation['task_state']['Position'] = assistant_action

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
