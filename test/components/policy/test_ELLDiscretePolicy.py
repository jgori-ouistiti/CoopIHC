import numpy
from coopihc.policy.ELLDiscretePolicy import ELLDiscretePolicy
from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, array_element, cat_element


policy = None


def test_init():
    global policy
    _seed = 123
    se = cat_element(N=7, seed=_seed)
    action_state = State(**{"action": se})
    policy = ELLDiscretePolicy(action_state, seed=_seed)
    assert policy.action_state is action_state
    assert policy.rng.uniform() == numpy.random.default_rng(_seed).uniform()


def compute_likelihood(self, action, observation, *args, **kwargs):
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


def test_attach_likelihood_function():
    global policy
    assert not hasattr(policy, "compute_likelihood")
    policy.attach_likelihood_function(compute_likelihood)

    assert hasattr(policy, "compute_likelihood")
    # Verify the method is properly bound (A bound method defines __self__)
    assert getattr(policy.compute_likelihood, "__self__", None) is not None
    probs = [
        1 / 7,
        1 / 7 + 0.05,
        1 / 7 - 0.05,
        1 / 7 + 0.1,
        1 / 7 - 0.1,
        1 / 7 + 0.075,
        1 / 7 - 0.075,
    ]
    for i in range(7):
        action = discrete_array_element(init=i, low=0, high=6)
        observation = {}
        assert policy.compute_likelihood(action, observation) == probs[i]


def test_forward_summary():
    global policy
    observation = {}
    actions, llh = policy.forward_summary(observation)
    assert abs(1 - sum(llh)) < 1e-13
    assert actions == [numpy.array(i) for i in range(7)]
    assert llh == [
        1 / 7,
        1 / 7 + 0.05,
        1 / 7 - 0.05,
        1 / 7 + 0.1,
        1 / 7 - 0.1,
        1 / 7 + 0.075,
        1 / 7 - 0.075,
    ]


def test_sample():
    global policy
    actions = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}
    for i in range(100000):
        if not i % 10000:
            print(i)
        action, reward = policy.sample(agent_observation={})
        actions[str(action.squeeze().tolist())] += 1
    empirical_probs = numpy.array([u / 100000 for u in list(actions.values())])
    llh = numpy.array(
        [
            1 / 7,
            1 / 7 + 0.05,
            1 / 7 - 0.05,
            1 / 7 + 0.1,
            1 / 7 - 0.1,
            1 / 7 + 0.075,
            1 / 7 - 0.075,
        ]
    )
    assert numpy.linalg.norm((empirical_probs - llh)) < 0.01


if __name__ == "__main__":
    test_init()
    test_attach_likelihood_function()
    test_forward_summary()
    test_sample()
