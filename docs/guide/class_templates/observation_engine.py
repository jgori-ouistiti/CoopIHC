class MyObservationEngine(BaseObservationEngine):
    def __init__(self, arg1, *args, **kwargs):
        super().__init__()
        self.type = 'enter type'
        self.arg1 = arg1

        _arg1 = {'value': arg1, 'meaning': 'meaning of arg1'}
        self.handbook['parameters'].extend([_arg1])

    def __content__(self):
        return _str

    def observe(self, game_state):
        game_state = copy.deepcopy(game_state)
        rewards = 0
        return game_state, rewards
