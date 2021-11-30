from coopihc.observation.BaseObservationEngine import BaseObservationEngine
import copy


class CascadedObservationEngine(BaseObservationEngine):
    def __init__(self, engine_list):
        super().__init__()
        self.engine_list = engine_list
        # self.type = "multi"

    def __content__(self):
        return {
            self.__class__.__name__: {
                "Engine{}".format(ni): i.__content__()
                for ni, i in enumerate(self.engine_list)
            }
        }

    def observe(self, game_state):
        game_state = copy.deepcopy(game_state)
        rewards = 0
        for engine in self.engine_list:
            new_obs, new_reward = engine.observe(game_state)
            game_state.update(new_obs)
            rewards += new_reward

        return game_state, rewards
