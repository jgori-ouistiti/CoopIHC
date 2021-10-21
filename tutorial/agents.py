from core.agents import BaseAgent
from core.observation import RuleObservationEngine, BaseUserObservationRule
from core.inference import InferenceEngine


import gym


## ========================= BaseAgent ================================================

### Initialize a BaseAgent as user
action_set = [[-1,1]]
action_space = [gym.spaces.Discrete(2)]
observation_engine = RuleObservationEngine(BaseUserObservationRule)
inference_engine = InferenceEngine(buffer_depth=0)
my_user = BaseAgent('user', action_space, action_set, observation_engine = observation_engine, inference_engine = inference_engine)

### Initialize a BaseAgent as user with a more complex action space
action_set = [[-1,1], None]
action_space = [gym.spaces.Discrete(2), gym.spaces.Box(low = -1, high = 1, shape = (1,))]
observation_engine = RuleObservationEngine(BaseUserObservationRule)
inference_engine = InferenceEngine(buffer_depth=0)
my_user = BaseAgent('user', action_space, action_set, observation_engine = observation_engine, inference_engine = inference_engine)

## Acces some attributes, sample the random policy
my_user.action_set
my_user.sample()
my_user.role
my_user.state

# Append a discrete and a continuous substate to the internal state
# Specify possible values that the state can take in the discrete case
my_user.append_state('Goal', [gym.spaces.Discrete(2)], possible_values = [-1,1])
# Don't specify possible values of the state for the continuous case
my_user.append_state('Courage', [gym.spaces.Box(low=-1, high=1, shape=(1, ))])


# See what is inside the state
my_user.state
my_user.state_dict
my_user.state_space

# Modify some substate
my_user.modify_state('Courage', value = 1)

## ======================= Subclassing BaseAgent ======================================
