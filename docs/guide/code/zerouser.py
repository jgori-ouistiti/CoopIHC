import core
from core.agents import BaseAgent
from core.policy import BasePolicy




class ExamplePolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        self.observation['user_state']['goal']

class NoisyPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExampleUser(BaseAgent):
    """ A half-random, half goal-oriented agent.
    """
    def __init__(self, *args, **kwargs):



        agent_policy =

        observation_engine =

        inference_engine =

        state =

        super().__init__(
            'user',
            'agent_policy' = agent_policy,
            'agent_observation_engine' = observation_engine,
            'agent_inference_engine' = inference_engine,
            'agent_state' = state,
            **kwargs    )


    def finit(self):
        ## write finit code here

    def reset(self, dic = None):
        if dic is None:
            super().reset()

        ## write reset code here.

        if dic is not None:
            super().reset(dic = dic)

    def render(self, *args, **kwargs):
        mode = kwargs.get('mode')
        if mode is None:
            mode = 'text'

        ## write render code here
