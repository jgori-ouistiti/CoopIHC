from copy import copy
from core.inference import BaseInferenceEngine


class RWInference(BaseInferenceEngine):
    def infer(self):
        if self.host.role == "user" and self.host.bundle.task.round > 0:
            try:
                # Get last choice and associated reward
                last_choice = self.host.action.values[0][0][0]
                last_task_state = self.buffer[-1]["task_state"]
                last_reward = last_task_state["last_reward"]["values"][0][0][0]

                # Get Q-values that led to that choice
                state = self.buffer[-1]["user_state"]
                q_values = state["q_values"]["values"][0][0]

                # Calculate error
                err = last_reward - q_values[last_choice]

                # Update Q-values based on choice and reward
                q_values[last_choice] += self.host.q_alpha * err

                # Update internal state and return it
                state["q_values"]["values"] = q_values

                return state, 0
            except KeyError:
                return super().infer()
        else:
            return super().infer()
