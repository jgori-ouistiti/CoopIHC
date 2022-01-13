from coopihc.observation.BaseObservationEngine import BaseObservationEngine
import copy


class CascadedObservationEngine(BaseObservationEngine):
    """CascadedObservationEngine

    Cascades (serially) several observation engines.

    Gamestate --> Engine1 --> Engine2 --> ... --> EngineN --> Observation


    :param engine_list: list of observation engines
    :type engine_list: list(:py:mod:`Observation Engine<coopihc.observation>`)
    """

    def __init__(self, engine_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_list = engine_list

    def __content__(self):
        """__content__

        Custom class repr

        :return: custom repr
        :rtype: dictionnary
        """
        return {
            self.__class__.__name__: {
                "Engine{}".format(ni): i.__content__()
                for ni, i in enumerate(self.engine_list)
            }
        }

    def observe(self, game_state):
        """observe

        Serial observations (i.e. output of an engine becomes input of the next one)

        :param game_state: game state
        :type game_state: `State<coopihc.space.State.State`
        :return: (observation, obs reward)
        :rtype: tuple(`State<coopihc.space.State.State`, float)
        """
        game_state = copy.deepcopy(game_state)
        rewards = 0
        for engine in self.engine_list:
            new_obs, new_reward = engine.observe(game_state)
            game_state.update(new_obs)
            rewards += new_reward

        return game_state, rewards
