import core
from core.agents import BaseAgent
from core.policy import BasePolicy

import numpy
from core.space import Space, StateElement, State
gamestate = State()
gamestate['task_state'] = State()
gamestate['user_state'] = State()
gamestate['task_state']['x'] = StateElement(values = numpy.array([0]), spaces = [Space([numpy.array([-4,-3,-2,-1,0,1,2,3,4], dtype = numpy.int16)])])
gamestate['user_state']['goal'] = StateElement(
    values = numpy.array([4]),
    spaces = [Space([numpy.array([-4,-3,-2,-1,0,1,2,3,4], dtype = numpy.int16)])]
                )


class ExamplePolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self):
        if self.observation['task_state']['x'] < self.observation['user_state']['goal']:
            _action_value = 1
        elif self.observation['task_state']['x'] > self.observation['user_state']['goal']:
            _action_value = -1
        else:
            _action_value = 0

        new_action = self.new_action['values'] = numpy.array(_action_value)
        return new_action, 0



class NoisyPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExampleUser(BaseAgent):
    """ A half-random, half goal-oriented agent.
    """
    def __init__(self, *args, **kwargs):

        # Defining State
        state = State()
        state["goal"] = StateElement(
            values = numpy.array([4]),
            spaces = [Space([numpy.array([-4,-3,-2,-1,0,1,2,3,4], dtype = numpy.int16)])]
                        )

        # Defining policy
        action_state = State()
        action_state['action'] = StateElement(
            values = None,
            spaces = [Space([numpy.array([-1,0,1], dtype = numpy.int16)])]
            )
        agent_policy = ExamplePolicy(action_state)

        observation_engine = None

        inference_engine = None



        super().__init__(
            'user',
            agent_policy = agent_policy,
            agent_observation_engine = observation_engine,
            agent_inference_engine = inference_engine,
            agent_state = state,
            **kwargs    )



from zerotask import ExampleTask
from core.bundle import SinglePlayUser
example_task = ExampleTask()
example_user = ExampleUser()
bundle = SinglePlayUser(example_task, example_user)
bundle.reset()
print(bundle.game_state)
bundle.step(bundle.user.policy.sample()[0])
print(bundle.game_state)
