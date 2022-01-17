from coopihc.policy.ExamplePolicy import ExamplePolicy
from coopihc.space.StateElement import StateElement
from coopihc.space.State import State
from coopihc.policy.ELLDiscretePolicy import ELLDiscretePolicy
from coopihc.space.utils import autospace


## ==================== ExamplePolicy
ep = ExamplePolicy()


## ==================== ELLDiscretePolicy

# [start-elld-def-model]
# Define the likelihood model
def likelihood_model(self, action, observation, *args, **kwargs):
    if action == 0:
        return 1 / 7
    elif action == 1:
        return 1 / 7 + 0.05
    elif action == 2:
        return 1 / 7 - 0.05
    elif action == 3:
        return 1 / 7 + 0.1
    elif action == 4:
        return 1 / 7 - 0.1
    elif action == 5:
        return 1 / 7 + 0.075
    elif action == 6:
        return 1 / 7 - 0.075
    else:
        raise RuntimeError(
            "warning, unable to compute likelihood. You may have not covered all cases in the likelihood definition"
        )


# [end-elld-def-model]

# [start-elld-attach]
_seed = 123
se = StateElement(1, autospace([0, 1, 2, 3, 4, 5, 6]), seed=_seed)
action_state = State(**{"action": se})
policy = ELLDiscretePolicy(action_state, seed=_seed)
# Attach the model
policy.attach_likelihood_function(likelihood_model)
# [end-elld-attach]
