from collections import OrderedDict
from core.inference import BaseInferenceEngine


class RWInference(BaseInferenceEngine):
    def infer(self):
        if self.host.role == "operator":
            try:
                last_choice = self.host.action.values[0]
                last_reward = self.buffer[-1]["task_state"]["last_reward"]["values"][0]

                state = self.host.state
                q_values = state["q_values"]["values"][0]

                err = last_reward - q_values[last_choice]
                q_values[last_choice] += self.host.q_alpha * err

                state["q_values"]["values"] = [q_values]

                return state, 0
            except KeyError:
                return super().infer()
        else:
            return super().infer()
