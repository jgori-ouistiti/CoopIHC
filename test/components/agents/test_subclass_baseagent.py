from coopihc.agents.BaseAgent import BaseAgent
from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.policy.ExamplePolicy import ExamplePolicy
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.inference.ExampleInferenceEngine import ExampleInferenceEngine
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine
from coopihc.observation.BaseObservationEngine import BaseObservationEngine
from coopihc.observation.ExampleObservationEngine import ExampleObservationEngine


import numpy


class MinimalAgent(BaseAgent):
    """Non-functional minimal subclass to use in tests."""


class NonMinimalAgent(BaseAgent):
    def __init__(self, *args, **kwargs):

        # custom policy
        action_state = State()
        action_state["action"] = discrete_array_element(init=2, low=1, high=3)
        policy = ExamplePolicy(action_state=action_state)

        # custom state

        state = State(**{"substate_1": cat_element(N=2, init=1)})

        # custom inference_engine

        inference_engine = ExampleInferenceEngine(buffer_depth=7)

        # custom observation_engine

        observation_engine = ExampleObservationEngine("substate_1")

        super().__init__(
            "user",
            *args,
            agent_state=state,
            agent_policy=policy,
            agent_inference_engine=inference_engine,
            agent_observation_engine=observation_engine,
            **kwargs
        )


def test_imports():
    """Tests the different import ways for the BaseAgent."""
    from coopihc import BaseAgent
    from coopihc.agents import BaseAgent
    from coopihc.agents.BaseAgent import BaseAgent


def test_example():
    """Tries to import and create the example user."""
    from coopihc.agents.ExampleUser import ExampleUser

    user = ExampleUser()
    return True


def test_init():
    """Tries to initialize an BaseAgent and checks the expected
    properties and methods."""
    test_properties()
    test_methods()


def test_properties():
    """Tests the expected properties for a minimal BaseAgent."""
    user = MinimalAgent("user")
    # Direct attributes
    assert hasattr(user, "bundle")
    assert hasattr(user, "ax")
    assert hasattr(user, "role")
    # components
    assert hasattr(user, "state")
    assert hasattr(user, "policy")
    assert hasattr(user, "observation_engine")
    assert hasattr(user, "inference_engine")
    # properties
    assert hasattr(user, "observation")
    assert hasattr(user, "action")


def test_methods():
    """Tests the expected methods for a minimal Interactionuser."""
    user = MinimalAgent("user")
    # Public methods
    assert hasattr(user, "finit")
    assert hasattr(user, "_attach_policy")
    assert hasattr(user, "_attach_observation_engine")
    assert hasattr(user, "_attach_inference_engine")
    assert hasattr(user, "reset")
    assert hasattr(user, "render")
    assert hasattr(user, "observe")

    # Private methods
    assert hasattr(user, "__content__")
    assert hasattr(user, "_base_reset")
    assert hasattr(user, "_override_components")
    assert hasattr(user, "take_action")
    assert hasattr(user, "_agent_step")


def test_minimalagent():
    """Tests the methods provided by the BaseAgent class."""
    test_imports()
    test_example()
    test_init()


def test_nonminimalagent():
    test_state()
    test_policy()
    test_inference_engine()
    test_observation_engine()


def test_state():
    agent = NonMinimalAgent()
    assert agent.state["substate_1"] == cat_element(N=2, init=1)


def test_policy():
    agent = NonMinimalAgent()
    assert isinstance(agent.policy, ExamplePolicy)
    assert agent.action == discrete_array_element(init=2, low=1, high=3)


def test_inference_engine():
    agent = NonMinimalAgent()
    assert isinstance(agent.inference_engine, ExampleInferenceEngine)
    assert agent.inference_engine.buffer_depth == 7


def test_observation_engine():
    agent = NonMinimalAgent()
    assert isinstance(agent.observation_engine, ExampleObservationEngine)
    assert agent.observation_engine.observable_state == "substate_1"


def test_override_components():
    test_override_components_args()
    test_override_components_kwargs()


def test_override_components_args():
    test_override_state()
    test_override_policy()
    test_override_obseng()
    test_override_infeng()


def test_override_components_kwargs():
    # test the mechanism when kwargs are provided
    return True


def test_override_state():
    state = State()
    agent = NonMinimalAgent(override_state=state)
    assert agent.state == State()


def test_override_policy():
    policy = BasePolicy()
    agent = NonMinimalAgent(override_policy=(policy, {}))
    assert isinstance(agent.policy, BasePolicy)
    assert agent.action == cat_element(N=2)


def test_override_obseng():
    obseng = BaseObservationEngine()
    agent = NonMinimalAgent(override_observation_engine=(obseng, {}))
    assert isinstance(agent.observation_engine, BaseObservationEngine)


def test_override_infeng():
    infeng = BaseInferenceEngine()
    agent = NonMinimalAgent(override_inference_engine=(infeng, {}))
    assert isinstance(agent.inference_engine, BaseInferenceEngine)
    assert agent.inference_engine.buffer_depth == 1


# +----------------------+
# +        MAIN          +
# +----------------------+
if __name__ == "__main__":
    test_minimalagent()
    test_nonminimalagent()
    test_override_components()
