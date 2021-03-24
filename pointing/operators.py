from core.agents import GoalDrivenDiscreteOperator
from core.models import BinaryOperatorModel
from core.observation import RuleObservationEngine, BaseOperatorObservationRule

class CarefulPointer(GoalDrivenDiscreteOperator):
    def __init__(self):
        operator_model = BinaryOperatorModel(1)
        observation_engine = RuleObservationEngine(BaseOperatorObservationRule)
        super().__init__(operator_model, observation_engine = observation_engine)
