import numpy
from coopihc.inference.GoalInferenceWithUserPolicyGiven import (
    GoalInferenceWithUserPolicyGiven,
)
from coopihc.policy.ELLDiscretePolicy import ELLDiscretePolicy
from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, array_element

# ----- needed before testing engine
action_state = State()
action_state["action"] = discrete_array_element(init=0, low=-3, high=3)
user_policy = ELLDiscretePolicy(action_state=action_state)

ERROR_RATE = 0.05


def compute_likelihood(self, action, observation, error_rate=ERROR_RATE):
    # convert actions and observations
    action = action.squeeze().tolist()
    position = observation["task_state"]["position"].tolist()
    if action == -position:
        return 1 - error_rate
    else:
        return error_rate


# Attach likelihood function to the policy
user_policy.attach_likelihood_function(compute_likelihood)


def test_init():

    inference_engine = GoalInferenceWithUserPolicyGiven()
    assert inference_engine.user_policy_model is None
    inference_engine = GoalInferenceWithUserPolicyGiven(user_policy_model=user_policy)
    assert inference_engine.user_policy_model is user_policy


assistant_state = State()
assistant_state["beliefs"] = array_element(
    init=numpy.array([1 / 7 for i in range(7)]),
    low=numpy.zeros((7,)),
    high=numpy.ones((7,)),
    out_of_bounds_mode="silent",  # may go outside of [0,1] e.g. prior to renormalizing.
)

chosen_action = 1
user_action = State()
user_action["action"] = discrete_array_element(init=chosen_action, low=-3, high=3)

observation = State(**{"assistant_state": assistant_state, "user_action": user_action})
set_theta = [
    {("task_state", "position"): discrete_array_element(init=t, low=-3, high=3)}
    for t in [-3, -2, -1, 0, 1, 2, 3]
]


def test_infer():
    inference_engine = GoalInferenceWithUserPolicyGiven(user_policy_model=user_policy)
    inference_engine.attach_set_theta(set_theta)
    inference_engine.buffer = []
    inference_engine.buffer.append(observation)
    state, reward = inference_engine.infer()
    assert reward == 0
    prior = [1 / 7 for i in range(7)]
    posterior = [1 / 7 * ERROR_RATE for i in range(7)]
    posterior[2] = 1 / 7 * (1 - ERROR_RATE)
    posterior = [p / sum(posterior) for p in posterior]
    assert (
        numpy.linalg.norm(
            state["beliefs"].reshape(
                -1,
            )
            - numpy.array(posterior)
        )
        < 1e-6
    )


if __name__ == "__main__":
    test_init()
    test_infer()
