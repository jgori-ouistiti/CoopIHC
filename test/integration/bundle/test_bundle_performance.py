import numpy
from coopihc import (
    State,
    StateElement,
    Space,
    Bundle,
    ExampleTask,
    BaseAgent,
    BasePolicy,
    discrete_space,
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
    user_action_state["action"] = StateElement(0, discrete_space([-1, 0, 1]))

    assistant_action_state = State()
    assistant_action_state["action"] = StateElement(0, discrete_space([-1, 0, 1]))

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
    assert bundle.task.state["x"] == 0
    assert isinstance(bundle.task.state["x"], StateElement)
    bundle.reset()
    assert bundle.task.state["x"] == 0
    assert isinstance(bundle.task.state["x"], StateElement)
    bundle.step(user_action=1, assistant_action=1)
    assert bundle.task.state["x"] == 2
    assert isinstance(bundle.task.state["x"], StateElement)


# +----------------------+
# +        MAIN          +
# +----------------------+
if __name__ == "__main__":
    cProfile.run("bundle_round()", sort="cumulative")
