import core
from core.agents import BaseAgent
from core.policy import BasePolicy,  BIGDiscretePolicy
from core.inference import GoalInferenceWithUserPolicyGiven
from core.space import State, StateElement, Space

import numpy
import copy

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
        action_space = Space([
                numpy.array([self.gain for i in range(self.bundle.task.dim)], dtype = numpy.float32),
                numpy.array([self.gain for i in range(self.bundle.task.dim)], dtype = numpy.float32)
                ])
        self.policy.action_state['action'] = StateElement(
            values = None,
            spaces = action_space,
            clipping_mode = 'clip')

class BIGGain(BaseAgent):
    def __init__(self):

        super().__init__(       'assistant',
                                inference_engine = GoalInferenceWithUserPolicyGiven() #
                                )


    def finit(self):
        action_state = self.bundle.game_state['assistant_action']
        action_state['action'] = StateElement(
            values = None,
            spaces = Space([
            numpy.array([i for i in range(self.bundle.task.gridsize)], dtype = numpy.int16)
                ]),
            clipping_mode = 'error'
        )
        user_policy_model = copy.deepcopy(self.bundle.user.policy)



        agent_policy = BIGDiscretePolicy(       action_state,
                                                user_policy_model
                                                )

        self.attach_policy(agent_policy)
        # self.inference_engine.attach_policy(agent_policy.user_policy_model)
        self.inference_engine.attach_policy(user_policy_model)



    def reset(self, dic = None):
        if dic is None:
            super().reset()

        self.state['beliefs'] = StateElement(
            values = numpy.array([1/self.bundle.task.number_of_targets for i in range(self.bundle.task.number_of_targets)]),
            spaces = Space([
                    numpy.zeros((1,self.bundle.task.number_of_targets)),
                    numpy.ones((1,self.bundle.task.number_of_targets))
                ]),
            clipping_mode = 'error'
            )

        # change theta for inference engine

        set_theta = [{     ('user_state', 'goal'):
            StateElement(
                values = [t],
                spaces = Space([
                    numpy.array([list(range(self.bundle.task.gridsize))], dtype = numpy.int16)
                            ])
                        )
                    } for t in self.bundle.task.state['targets']['values']
                    ]

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
            axtask, axuser, axassistant = args
            self.inference_engine.render(axassistant, mode = mode)
        except ValueError:
            self.inference_engine.render(mode = mode)
