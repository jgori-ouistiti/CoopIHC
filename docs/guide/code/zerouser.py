import coopihc
from coopihc.agents import BaseAgent
from coopihc.policy import BasePolicy

import numpy
from coopihc.space import Space, StateElement, State

gamestate = State()
gamestate["task_state"] = State()
gamestate["user_state"] = State()
gamestate["task_state"]["x"] = StateElement(
    values=numpy.array([0]),
    spaces=[Space([numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16)])],
)
gamestate["user_state"]["goal"] = StateElement(
    values=numpy.array([4]),
    spaces=[Space([numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16)])],
)


class ExamplePolicy(BasePolicy):
    """A simple policy which assumes that the agent using it has a goal state and that the task has an 'x' state. x is compared to the goal and appropriate action is taken to make sure x reaches the goal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self):
        if (
            self.observation["task_state"]["x"]
            < self.observation["{}_state".format(self.host.role)]["goal"]
        ):
            _action_value = 1
        elif (
            self.observation["task_state"]["x"]
            > self.observation["{}_state".format(self.host.role)]["goal"]
        ):
            _action_value = -1
        else:
            _action_value = 0

        new_action = self.new_action["values"] = numpy.array(_action_value)
        return new_action, 0


class NoisyPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExampleUser(BaseAgent):
    """A half-random, half goal-oriented agent."""

    def __init__(self, *args, **kwargs):

        # Define an internal state with a 'goal' substate
        state = State()
        state["goal"] = StateElement(
            values=numpy.array([4]),
            spaces=[
                Space([numpy.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=numpy.int16)])
            ],
        )

        # Call the policy defined above
        action_state = State()
        action_state["action"] = StateElement(
            values=None,
            spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])],
        )
        agent_policy = ExamplePolicy(action_state)

        # Use default observation and inference engines
        observation_engine = None
        inference_engine = None

        super().__init__(
            "user",
            agent_policy=agent_policy,
            agent_observation_engine=observation_engine,
            agent_inference_engine=inference_engine,
            agent_state=state,
            **kwargs
        )

    # Override default behaviour of BaseAgent which would randomly sample new goal values on each reset. Here for purpose of demonstration we impose a goal = 4
    def reset(self, dic=None):
        self.state["goal"]["values"] = 4


from examples.tasks import ExampleTask


class ExampleTaskWithoutAssistant(ExampleTask):
    def assistant_step(self, *args, **kwargs):
        return self.state, 0, False, {}


from coopihc.bundle import Bundle

example_task = ExampleTaskWithoutAssistant()
example_user = ExampleUser()
bundle = Bundle(task=example_task, user=example_user)
bundle.reset(turn=1)
while 1:
    state, rewards, is_done = bundle.step(bundle.user.policy.sample()[0])
    print(state, rewards, is_done)
    if is_done:
        break
