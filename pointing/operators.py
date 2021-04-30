from core.agents import GoalDrivenDiscreteOperator
from core.models import GoalDrivenBinaryOperatorModel
from core.observation import RuleObservationEngine, base_operator_engine_specification

import numpy
from collections import OrderedDict

class CarefulPointer(GoalDrivenDiscreteOperator):
    """ An operator that only indicates the right direction, with a fixed amplitude.

    Works with a task that has a 'Targets' substate. At each reset, it selects a new goal from the possible 'Targets'. When sampled, the operator will issue an action that is either +1 or -1 in the direction of the target.
    The operator observes everything perfectly except for the assistant state.

    A CarefulPointer has the following characteristics:

        * An operator model ``GoalDrivenBinaryOperatorModel(1)``
        * An observation engine ``RuleObservationEngine(BaseOperatorObservationRule)``
        * No inference

    :meta public:
    """
    def __init__(self):
        operator_model = GoalDrivenBinaryOperatorModel(1)
        observation_engine = RuleObservationEngine(base_operator_engine_specification)
        super().__init__(operator_model, observation_engine = observation_engine)






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
