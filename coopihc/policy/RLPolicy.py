import copy

from coopihc.space.State import State
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.bundle.wrappers.Train import TrainGym

# ======================= RL Policy


class RLPolicy(BasePolicy):
    """Wrap a trained net as a CoopIHC policy.

    A policy object compatible with CoopIHC that wraps a policy that was trained via Deep Reinforcement learning.

    Example code:

    .. code-block:: python

        # action_state
        action_state = State()
        action_state["action"] = StateElement(0, autospace([-5 + i for i in range(11)]))

        # env
        env = TrainGym(
        bundle,
        train_user=True,
        train_assistant=False,
            )

        # Using PPO from stable_baselines3, with some wrappers
        model_path = "saved_model.zip"
        learning_algorithm = "PPO"
        wrappers = {
            "observation_wrappers": [MyObservationWrapper],
            "action_wrappers": [MyActionWrapper],
        }
        library = "stable_baselines3"

        trained_policy = RLPolicy(
            action_state, model_path, learning_algorithm, env, wrappers, library
        )


    .. note ::

        Currently only supports policies obtained via stable baselines 3.





    :param action_state: see ``BasePolicy``
    :type action_state: see ``BasePolicy``
    :param model_path: path to the saved model
    :type model_path: string
    :param learning_algorithm: name of the learning algorithm
    :type learning_algorithm: string
    :param env: environment before any wrappers were applied
    :type env: gym.Env
    :param wrappers: observation and action wrappers
    :type wrappers: dictionary
    :param library: name of the training library. Currently, only stable_baselines3 is supported.
    :type library: string
    """

    def __init__(
        self,
        action_state,
        model_path,
        learning_algorithm,
        env,
        wrappers,
        library,
        *args,
        **kwargs
    ):
        model_path = model_path
        self.learning_algorithm = learning_algorithm
        self.env = env
        self.obs_wraps = wrappers["observation_wrappers"]
        self.act_wraps = wrappers["action_wrappers"]
        self.library = library

        # self.wrappers = kwargs.get("wrappers")

        if library != "stable_baselines3":
            raise NotImplementedError(
                "The Reinforcement Learning Policy currently only supports policies obtained via stable baselines 3."
            )
        import stable_baselines3

        learning_algorithm = getattr(stable_baselines3, learning_algorithm)
        self.model = learning_algorithm.load(model_path)

        # Recovering action space

        super().__init__(*args, action_state=action_state, **kwargs)

    def sample(self, observation=None):
        """sample

        Get action by using model.predict(deterministic = True), applying the necessary wrappers.

        :param observation: see ``BasePolicy``
        :type observation: see ``BasePolicy``, optional
        :return: see ``BasePolicy``
        :rtype: see ``BasePolicy``
        """
        if observation is None:
            observation = self.observation

        # convert observation via the Train class
        observation = self.env._convertor.filter_gamestate(
            observation, self.env.observation_mapping
        )

        # Apply observation Wrappers
        for w in self.obs_wraps:
            observation = w.observation(w, observation)

        action = self.model.predict(observation, deterministic=True)[
            0
        ]  # with deterministic = True, don't sample from the Gaussian but just take its mean

        # Apply Action Wrappers
        for w in self.act_wraps:
            action = w.action(w, action)

        # convert action via the Train class
        action = list(
            self.env._convertor.adapt_discrete_and_multidiscrete_action(
                action, self.env
            ).values()
        )

        new_action = self.new_action
        new_action[:] = action
        return new_action, 0
