from core.agents import BaseAgent
from core.observation import RuleObservationEngine, BaseOperatorObservationRule
from core.inference import InferenceEngine


import gym


## ========================= BaseAgent ================================================

### Initialize a BaseAgent as operator
action_set = [[-1,1]]
action_space = [gym.spaces.Discrete(2)]
observation_engine = RuleObservationEngine(BaseOperatorObservationRule)
inference_engine = InferenceEngine(buffer_depth=0)
my_operator = BaseAgent('operator', action_space, action_set, observation_engine = observation_engine, inference_engine = inference_engine)

### Initialize a BaseAgent as operator with a more complex action space
action_set = [[-1,1], None]
action_space = [gym.spaces.Discrete(2), gym.spaces.Box(low = -1, high = 1, shape = (1,))]
observation_engine = RuleObservationEngine(BaseOperatorObservationRule)
inference_engine = InferenceEngine(buffer_depth=0)
my_operator = BaseAgent('operator', action_space, action_set, observation_engine = observation_engine, inference_engine = inference_engine)

## Acces some attributes, sample the random policy
my_operator.action_set
my_operator.sample()
my_operator.role
my_operator.state

# Append a discrete and a continuous substate to the internal state
# Specify possible values that the state can take in the discrete case
my_operator.append_state('Goal', [gym.spaces.Discrete(2)], possible_values = [-1,1])
# Don't specify possible values of the state for the continuous case
my_operator.append_state('Courage', [gym.spaces.Box(low=-1, high=1, shape=(1, ))])


# See what is inside the state
my_operator.state
my_operator.state_dict
my_operator.state_space

# Modify some substate
my_operator.modify_state('Courage', value = 1)

## ======================= Subclassing BaseAgent ======================================
