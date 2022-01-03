import numpy
from coopihc import (
    State,
    StateElement,
    Space,
    Bundle,
    ExampleTask,
    BaseAgent,
    BasePolicy,
)
import cProfile


def test_bundle_round(benchmark):
    """Runs a performance test for a single bundle round using
    pytest-benchmark."""
    benchmark(bundle_round)


def bundle_round():
    """Runs a single bundle round using the ExampleTask and two
    BaseAgents as user and assistant."""
    # Define agent action states (what actions they can take)
    user_action_state = State()
    user_action_state["action"] = StateElement(
        values=None,
        spaces=[Space([numpy.array([-1, 0, 1], dtype=int)])],
    )

    assistant_action_state = State()
    assistant_action_state["action"] = StateElement(
        values=None,
        spaces=[Space([numpy.array([-1, 0, 1], dtype=int)])],
    )

    # Bundle a task together with two BaseAgents
    bundle = Bundle(
        task=ExampleTask(),
        user=BaseAgent("user", override_agent_policy=BasePolicy(user_action_state)),
        assistant=BaseAgent(
            "assistant",
            override_agent_policy=BasePolicy(assistant_action_state),
        ),
    )

    # Reset the task, run a round
    bundle.reset(turn=1)
    assert bundle.task.state["x"].values[0][0][0] == 0
    bundle.step(user_action=1, assistant_action=1)
    assert bundle.task.state["x"].values[0][0][0] == 2


# +----------------------+
# +        MAIN          +
# +----------------------+
if __name__ == "__main__":
    cProfile.run("bundle_round()", sort="cumulative")
