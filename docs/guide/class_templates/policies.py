class MyNewPolicy(Policy):
    """ Use this class as template to build your new policy.

    :param myparameter(type): explain the parameter here
    :return: what is returned
    :meta public:
    """
    def __init__(self, arg1, *args, **kwargs):
        action_state =
        super().__init__(action_state, *args, **kwargs)
        self.arg1 = arg1
        self.handbook['render_mode'].extend(['plot', 'text', 'log'])
        _arg1 = {'value': arg1, 'meaning': 'meaning of arg1'}
        self.handbook['parameters'].extend([_arg1])


    def sample(self):
        action =
        reward =
        return action, reward

    def reset(self):
        pass
