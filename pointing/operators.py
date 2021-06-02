from core.agents import BaseAgent
from core.observation import RuleObservationEngine, base_operator_engine_specification
from core.space import State, StateElement
from core.policy import ELLDiscretePolicy
import gym


class CarefulPointer(BaseAgent):
    """ An operator that only indicates the right direction, with a fixed amplitude.

    Works with a task that has a 'Targets' substate. At each reset, it selects a new goal from the possible 'Targets'. When sampled, the operator will issue an action that is either +1 or -1 in the direction of the target.
    The operator observes everything perfectly except for the assistant state.


    :meta public:
    """
    def __init__(self, **kwargs):

        # --------- Defining the agent's policy ----------
        # Here we consider a simulated user, which will only indicate left or right (assumed to be in the right direction of the target 99% of the time)

        agent_policy = kwargs.get('agent_policy')
        if agent_policy is None:
            agent_policy = ELLDiscretePolicy(action_space = [gym.spaces.Discrete(2)], action_set = [[-1, 1]])

            # Actions are in human values, i.e. they are not necessarily in range(0,N)
            def compute_likelihood(self, action, observation):
                # convert actions and observations
                action = action['human_values'][0]
                goal = observation['operator_state']['Goal']['human_values'][0]
                position = observation['task_state']['Position']['human_values'][0]

                # Write down all possible cases (5)
                # (1) Goal to the right, positive action
                if goal > position and action > 0 :
                    return .99
                # (2) Goal to the right, negative action
                elif goal > position and action < 0 :
                    return .01
                # (3) Goal to the left, positive action
                if goal < position and action > 0 :
                    return .01
                # (4) Goal to the left, negative action
                elif goal < position and action < 0 :
                    return .99
                elif goal == position:
                    return 0
                else:
                    raise RunTimeError("warning, unable to compute likelihood. You may have not covered all cases in the likelihood definition")

            # Attach likelihood function to the policy
            agent_policy.attach_likelihood_function(compute_likelihood)



        # ---------- Observation engine ------------
        # High-level specification
        observation_engine = kwargs.get('observation_engine')

        if observation_engine is None:
            base_operator_engine_specification  =    [ ('turn_index', 'all'),
                                                ('task_state', 'all'),
                                                ('operator_state', 'all'),
                                                ('assistant_state', None),
                                                ('operator_action', 'all'),
                                                ('assistant_action', 'all')
                                                ]
            # Additional deterministic and probabilistic 'rules' that can be added to the engine: for example, to add noise to a component, or to target one component in particular
            extradeterministicrules = {}
            extraprobabilisticrules = {}
            observation_engine = RuleObservationEngine(
                    deterministic_specification = base_operator_engine_specification,
                    extradeterministicrules = extradeterministicrules,
                    extraprobabilisticrules = extraprobabilisticrules   )

        # ---------- Calling BaseAgent class -----------
        # Calling an agent, set as an operator, which uses our previously defined observation engine and without an inference engine.


        super().__init__(
                            'operator',
                            policy = agent_policy,
                            observation_engine = observation_engine,
                            inference_engine = None)



    def finit(self):
        target_values = self.bundle.task.state['Targets']['values']
        target_spaces = self.bundle.task.state['Targets']['spaces']
        self.state['Goal'] =  StateElement( values = None,
                                            spaces = [gym.spaces.Discrete(len(target_spaces))],
                                            possible_values = [target_values])


    def reset(self, *args):
        self.finit()
        super().reset(*args)



# class LQGPointer(LQG_SS):
#     def __init__(self, dim, depth,  *args):
#         self.dim = dim
#         self.depth = depth
#         self.timestep = 0.1
#         if depth == 4:
#             I, b, ta, te = args
#         else:
#             raise NotImplementedError
#         a1 = b/(ta*te*I)
#         a2 = 1/(ta*te) + (1/ta + 1/te)*b/I
#         a3 = b/I + 1/ta + 1/te
#         bu = 1/(ta*te*I)
#
#         A = numpy.array([   [0, 1, 0, 0],
#                         [0, 0, 1, 0],
#                         [0, 0, 0, 1],
#                         [0, -a1, -a2, -a3]    ])
#
#         B = numpy.array([[ 0, 0, 0, bu]]).reshape((-1,1))
#
#         C = numpy.array([   [1, 0, 0, 0],
#                             [0, 1, 0, 0],
#                             [0, 0, 1, 0]
#                                 ])
#
#
#
#         D = numpy.array([   [0.01, 0, 0],
#                             [0, 0.01, 0],
#                             [0, 0, 0.05]
#                             ])
#
#         F = numpy.diag([0, 0, 0, 0.001])
#         Gamma = numpy.array([[0.08]])
#         G = 0.03*numpy.diag([1,1,0,0])
#
#         Q = numpy.diag([1, 0.01, 0, 0])
#         R = numpy.array([[1e-3]])
#         U = numpy.diag([1, 0.1, 0.01, 0])
#
#
#         D = D*0.35
#         G = G*0.35
#
#         super().__init__('operator', A, B, C, D, F, G, Q, R, U, Gamma)
#         self.state = OrderedDict({'x': numpy.array([0 for i in range(self.depth*self.dim)]), 'xhat': numpy.array([0 for i in range(self.depth*self.dim)])})
#
#     def reset(self, *args):
#         if args:
#             print(args)
#             raise NotImplementedError
#         else:
#             # select starting position
#             x0 = -.5
#             # Start from still
#             x_array = [-.5] + [0 for i in (range(self.depth*self.dim-1))]
#
#             self.state = OrderedDict({'x': numpy.array(x_array), 'xhat': numpy.array([0 for i in range(self.depth*self.dim)])})
