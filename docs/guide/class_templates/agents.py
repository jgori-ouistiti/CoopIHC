from coopihc.agents import BaseAgent

class MyNewAgent(BaseAgent):
    """ Use this class as template to build your new agent.

    :param myparameter(type): explain the parameter here
    :return: what is returned
    :meta public:
    """
    def __init__(self, arg1, *args, **kwargs):
        self.arg1 = arg1


        agent_policy = kwargs.get('agent_policy')
        if agent_policy is None:
            # agent_policy =

        observation_engine = kwargs.get('observation_engine')
        if observation_engine is None:
            # observation_engine =

        inference_engine = kwargs.get('inference_engine')
        if inference_engine is None:
            # inference_engine =

        state = kwargs.get('state')
        if state is None:
            # state =

        super().__init__('user',
                            state = state,
                            policy = agent_policy,
                            observation_engine = observation_engine,
                            inference_engine = inference_engine
                            )


        self.handbook['render_mode'].extend(['plot', 'text', 'log'])
        _arg1 = {'value': arg1, 'meaning': 'meaning of arg1'}
        self.handbook['parameters'].extend([_arg1])


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
