from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine
from coopihc.bundle.Simulator import Simulator
import numpy
import copy

# [start-infeng-subclass]
class ExampleInferenceEngine(BaseInferenceEngine):
    """ExampleInferenceEngine

    Example class

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def infer(self, agent_observation=None):
        """infer

        Do nothing. Same behavior as parent ``BaseInferenceEngine``

        :return: (new internal state, reward)
        :rtype: tuple(:py:class:`State<coopihc.base.State.State>`, float)
        """
        if agent_observation is None:
            agent_state = self.state

        reward = 0
        # Do something
        # agent_state = ..
        # reward = ...

        return agent_state, reward


ExampleInferenceEngine(buffer_depth=5)
# [end-infeng-subclass]


class CoordinatedInferenceEngine(BaseInferenceEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def simulation_bundle(self):
        return self.host.simulation_bundle

    @BaseInferenceEngine.default_value
    def infer(self, agent_observation=None):

        agent_state = getattr(agent_observation, f"{self.role}_state")

        # Parameter Inference is naive on purpose here
        while True:
            # Prediction using user model (can sample directly from the policy in this case, because it already does a single-shot prediction)
            usermodel_action, _ = self.host.take_action(
                agent_observation=agent_observation, agent_state=self.state
            )

            # actual observation
            user_action = agent_observation.user_action.action

            # Compare prediction with observation
            if user_action != usermodel_action:
                # If different, increment parameter by 1 and apply modulo 10. This works because we assumed we knew everything except the value of this parameter.
                agent_state["user_p0"][...] = (agent_state["user_p0"] + 1) % 10
            else:
                break

        reward = 0

        return self.state, reward


class RolloutCoordinatedInferenceEngine(BaseInferenceEngine):
    def __init__(self, task_model, user_model, assistant, **kwargs):
        super().__init__(**kwargs)
        self.task_model = task_model
        self.user_model = user_model
        self.assistant = assistant
        self._simulator = None
        self.__inference_count = 0

    # define the simulator here. Simulator is called like a Bundle, but it will use the dual version of objects if available.
    @property
    def simulator(self):
        if self._simulator is None:
            self._simulator = Simulator(
                task_model=self.task_model,
                user_model=self.user_model,
                assistant=self.assistant,
            )
        return self._simulator

    @BaseInferenceEngine.default_value
    def infer(self, agent_observation=None):

        if (
            self.__inference_count > 0
        ):  # If it is the first time there is inference, continue, else just perform a BaseInference. We can do this because we know the parameter p[0] will not evolve over time.
            return super().infer(agent_observation=agent_observation)

        self.__inference_count += 1

        agent_state = getattr(agent_observation, f"{self.role}_state")

        mem_state = copy.deepcopy(
            agent_state
        )  # agent state will be altered in the simulator, so keep a copy of it for reference.

        # For the 10 possible values, completely simulate them. The right parameter is the one that leads to the maximum rewards
        rew = [0 for i in range(10)]
        for i in range(10):  # Exhaustively try out all cases

            # load the simulation with the right parameters
            reset_dic = copy.deepcopy(agent_observation)

            # try out a new state
            del reset_dic["assistant_state"]
            reset_dic = {
                **reset_dic,
                **{
                    "assistant_state": {
                        "p0": i,
                        "p1": mem_state.p1,
                        "p2": mem_state.p2,
                    }
                },
            }

            self.simulator.reset(go_to=0, dic=reset_dic)
            while True:
                state, rewards, is_done = self.simulator.step()
                rew[i] += sum(rewards.values())
                if is_done:
                    break

        # Don't forget to close the simulator when you are finished.
        self.simulator.close()

        index = numpy.argmax(rew)
        self.state["user_p0"] = index
        return mem_state, 0
