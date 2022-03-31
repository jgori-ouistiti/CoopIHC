import numpy
from coopihc.agents.BaseAgent import BaseAgent
from coopihc.base.State import State
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.base.elements import cat_element, discrete_array_element
from coopihc.policy.ExamplePolicy import ExamplePolicy, PseudoRandomPolicy


class ExampleUser(BaseAgent):
    """An Example of a User.

    An agent that handles the ExamplePolicy, has a single 1d state, and has the default observation and inference engines.
    See the documentation of the :py:mod:`BaseAgent <coopihc.agents.BaseAgent.BaseAgent>` class for more details.

    :meta public:
    """

    def __init__(self, *args, **kwargs):

        # Define an internal state with a 'goal' substate
        state = State()
        state["goal"] = discrete_array_element(init=4, low=-4, high=4)

        # Define policy
        action_state = State()
        action_state["action"] = discrete_array_element(init=0, low=-1, high=1)
        agent_policy = ExamplePolicy(action_state=action_state)

        # Use default observation and inference engines
        observation_engine = None
        inference_engine = None

        super().__init__(
            "user",
            *args,
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            agent_state=state,
            **kwargs
        )

    def reset(self, dic=None):
        """reset

        Override default behaviour of BaseAgent which would randomly sample new goal values on each reset. Here for purpose of demonstration we impose a goal = 4

        :meta public:
        """
        self.state["goal"] = 4


class PseudoRandomUser(BaseAgent):
    def __init__(self, *args, **kwargs):

        # Define an internal state with a 'goal' substate
        state = State()
        state["p0"] = discrete_array_element(init=1, low=-10, high=10)
        state["p1"] = discrete_array_element(init=5, low=-10, high=10)
        state["p2"] = discrete_array_element(init=7, low=-10, high=10)

        # Call the policy defined above
        action_state = State()
        action_state["action"] = discrete_array_element(init=0, N=10)
        agent_policy = PseudoRandomPolicy(action_state=action_state)

        # Use default observation and inference engines
        observation_engine = None
        inference_engine = None

        super().__init__(
            "user",
            *args,
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            agent_state=state,
            **kwargs
        )


class PseudoRandomUserWithParams(BaseAgent):
    def __init__(self, p=[1, 1, 1], *args, **kwargs):

        # Define an internal state with a 'goal' substate
        self.p = p
        state = State()
        state["p0"] = discrete_array_element(init=p[0], low=-10, high=10)
        state["p1"] = discrete_array_element(init=p[1], low=-10, high=10)
        state["p2"] = discrete_array_element(init=p[2], low=-10, high=10)

        # Call the policy defined above
        action_state = State()
        action_state["action"] = discrete_array_element(init=0, N=10)
        agent_policy = PseudoRandomPolicy(action_state=action_state)

        # Use default observation and inference engines
        observation_engine = None
        inference_engine = None

        super().__init__(
            "user",
            *args,
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            agent_state=state,
            **kwargs
        )
