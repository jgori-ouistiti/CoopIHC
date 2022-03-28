import numpy
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine
from coopihc.inference.CascadedInferenceEngine import CascadedInferenceEngine


from coopihc.base.elements import example_game_state

egstate = example_game_state()["user_state"]


class DummyInferenceEngine(BaseInferenceEngine):
    def infer(self, user_state=None):
        if user_state is None:
            user_state = self.state

        user_state["goal"] = user_state["goal"] + 1
        return user_state, 1


engine_list = [DummyInferenceEngine(), DummyInferenceEngine()]
inference_engine = CascadedInferenceEngine(engine_list)


def test_cascade_init():
    assert inference_engine.engine_list == engine_list
    assert egstate["goal"] == 0
    state, reward = inference_engine.infer(user_state=egstate)
    assert reward == 2
    assert state["goal"] == 2


if __name__ == "__main__":
    test_cascade_init()
