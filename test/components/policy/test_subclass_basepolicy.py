from coopihc import BasePolicy, State, discrete_array_element


def test_init():
    class NewPolicy(BasePolicy):
        def __init__(self, action_state):
            super().__init__(action_state=action_state)

    action_state = State(**{"action": discrete_array_element(N=5)})

    new_policy = NewPolicy(action_state)
    assert new_policy.action_state.action.space.N == 5


if __name__ == "__main__":
    test_init()
