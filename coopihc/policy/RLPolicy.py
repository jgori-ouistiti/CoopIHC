import copy

from coopihc.space.State import State
from coopihc.policy.BasePolicy import BasePolicy

# ======================= RL Policy


class RLPolicy(BasePolicy):
    """Code works as proof of concept, but should be tested and augmented to deal with arbitrary wrappers. Possibly the wrapper class should be augmented with a reverse method, or something like that."""

    def __init__(self, *args, **kwargs):
        self.role = args[0]
        model_path = kwargs.get("model_path")
        learning_algorithm = kwargs.get("learning_algorithm")
        library = kwargs.get("library")
        self.training_env = kwargs.get("training_env")
        self.wrappers = kwargs.get("wrappers")

        if library != "stable_baselines3":
            raise NotImplementedError(
                "The Reinforcement Learning Policy currently only supports policies obtained via stables baselines 3."
            )
        import stable_baselines3

        learning_algorithm = getattr(stable_baselines3, learning_algorithm)
        self.model = learning_algorithm.load(model_path)

        # Recovering action space
        action_state = State()
        action_state["action"] = copy.deepcopy(
            getattr(
                getattr(
                    getattr(self.training_env.unwrapped.bundle, "user"),
                    "policy",
                ),
                "action_state",
            )["action"]
        )

        super().__init__(action_state, *args, **kwargs)

    def sample(self):
        # observation = self.host.inference_engine.buffer[-1]
        observation = self.observation
        nn_obs = self.training_env.unwrapped.convert_observation(observation)
        _action = self.model.predict(nn_obs)[0]
        for wrappers_name, (_cls, _args) in reversed(
            self.wrappers["actionwrappers"].items()
        ):
            aw = _cls(self.training_env.unwrapped, *_args)
            _action = aw.action(_action)
        action = self.action_state["action"]
        action["values"] = _action
        return action, 0
