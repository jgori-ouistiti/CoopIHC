from abc import ABC

from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import base_user_engine_specification
from coopihc.observation.utils import base_assistant_engine_specification
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


class BaseAgent(ABC):
    """Instantiate or subclass this class to define an agent that can be used in a Bundle.




    By default, this class will be initialized with an empty internal ``State``, a ``BasePolicy`` with a single action, a ``RuleObservationEngine`` with a ``BaseUser`` or ``BaseAssistant`` profile, and a ``BaseInference`` engine.

    Some things to know:

    * You can override some components, e.g. to override the existing policy of an agent named ``MyNewUser`` with some other policy, you can do the following

    .. code-block:: python

        changed_policy_user = MyNewUser(override_agent_policy = (some_other_policy, other_policy_kwargs))

    * You can access observations and actions of an agent directly via the built-in properties action and observation. For example, you access the last observation of the agent ``MyNewUser`` with

    .. code-block:: python

        last_obs = MyNewUser.observation

    The API methods that users of this class can redefine are:

        + finit (a second round of initialization once the bundle has been formed)
        + reset (to specify how to initialize the state of the agent at the end of each game)
        + render (specify display)


    :param str role: "user" or "assistant"
    :param type \*\*kwargs: keyword values ( each agent_X key expects a valid X object, and X_kwargs expects a valid dictionary of keyword arguments for X)

        + agent_policy
        + agent_inference_engine
        + agent_observation_engine
        + agent_state
        + policy_kwargs
        + inference_engine_kwargs
        + observation_engine_kwargs
        + state_kwargs


    :return: A :py:class:`Bundle<coopihc.bundle>`-compatible agent
    :rtype: BaseAgent

    """

    def __init__(
        self,
        role,
        agent_state=None,
        agent_policy=None,
        agent_inference_engine=None,
        agent_observation_engine=None,
        state_kwargs={},
        policy_kwargs={},
        inference_engine_kwargs={},
        observation_engine_kwargs={},
        *args,
        **kwargs
    ):

        # Bundles stuff
        self.bundle = None
        self.ax = None

        # Set role of agent
        if role not in ["user", "assistant"]:
            raise ValueError(
                "First argument 'role' should be either 'user' or 'assistant'"
            )
        else:
            self.role = role

        # Define policy
        self.attach_policy(agent_policy, **policy_kwargs)

        # Init state
        if agent_state is None:
            self.state = State(**state_kwargs)
        else:
            self.state = agent_state

        # Define observation engine
        self.attach_observation_engine(
            agent_observation_engine, **observation_engine_kwargs
        )

        # Define inference engine
        self.attach_inference_engine(agent_inference_engine, **inference_engine_kwargs)

        self._override_components(kwargs)

    def _override_components(self, init_kwargs):
        """_override_components

        Allows the end-user to override any component for any agent via a kwarg

        kwargs are as follows:

            * 'override_policy' = (policy, policy_kwargs)
            * 'override_state' = state
            * 'override_observation_engine' = (observation engine, observation engine_kwargs)
            * 'override_inference_engine' = (inference engine, inference engine kwargs)

        :param init_kwargs: kwargs passed from init
        :type init_kwargs: dict
        """
        # Override agent policy
        agent_policy, agent_policy_kwargs = init_kwargs.get(
            "override_policy", (None, None)
        )
        if agent_policy is not None:
            self.attach_policy(agent_policy, **agent_policy_kwargs)

        # Override agent state
        agent_state = init_kwargs.get("override_state", None)
        if agent_state is not None:
            self.state = agent_state

        # Override agent observation engine
        agent_obseng, agent_obseng_kwargs = init_kwargs.get(
            "override_observation_engine", (None, None)
        )
        if agent_obseng is not None:
            self.attach_observation_engine(agent_obseng, **agent_obseng_kwargs)

        # Override agent inference engine
        agent_infeng, agent_infeng_kwargs = init_kwargs.get(
            "override_inference_engine", (None, None)
        )
        if agent_infeng is not None:
            self.attach_inference_engine(agent_infeng, **agent_infeng_kwargs)

    def __content__(self):
        """Custom class representation.

        A custom representation of the class.

        :return: dictionary with content for all components.
        :rtype: dict

        :meta private:
        """
        return {
            "Name": self.__class__.__name__,
            "State": self.state.__content__(),
            "Observation Engine": self.observation_engine.__content__(),
            "Inference Engine": self.inference_engine.__content__(),
            "Policy": self.policy.__content__(),
        }

    @property
    def observation(self):
        """observation (property)

        Returns the last observation.


        :return: Last observation
        :rtype: coopihc.space.State.State

        :meta private:
        """
        return (
            self.inference_engine.buffer[-1]
            if self.inference_engine.buffer is not None
            else None
        )

    @property
    def action(self):
        """action (property)

        Returns the last action.


        :return: Last action
        :rtype: coopihc.space.State.State

        :meta private:
        """
        return self.policy.action_state["action"]

    def attach_policy(self, policy, **kwargs):
        """Attach a policy

        Helper function to attach a policy.

        :param policy: a CoopIHC policy
        :type policy: :doc:mod:`Policy<coopihc/policy>`

        :meta private:
        """
        if policy is None:
            policy = BasePolicy

        if type(policy).__name__ == "type":
            self.policy = policy(**kwargs)
        else:
            self.policy = policy
            if kwargs != {}:
                raise AttributeError(
                    "Can't input an instantiated policy and associated keyword arguments. Either pass the policy class, or fully instantiate that policy before passing it."
                )

        self.policy.host = self

    def attach_observation_engine(self, observation_engine, **kwargs):
        """Attach an observation engine

        Helper function to attach an observation engine.

        :param observation_engine: a CoopIHC observation engine
        :type observation_engine: :doc:mod:`Observation Engine<coopihc/observation>`

        :meta private:
        """
        if observation_engine is None:
            if self.role == "user":
                observation_engine = RuleObservationEngine(
                    deterministic_specification=base_user_engine_specification
                )
            elif self.role == "assistant":
                observation_engine = RuleObservationEngine(
                    deterministic_specification=base_assistant_engine_specification
                )
            else:
                raise NotImplementedError

        if type(observation_engine).__name__ == "type":
            self.observation_engine = observation_engine(**kwargs)
        else:
            self.observation_engine = observation_engine
            if kwargs != {}:
                raise AttributeError(
                    "Can't input an instantiated observation engine and associated keyword arguments. Either pass the observation engine class, or fully instantiate that policy before passing it."
                )

        self.observation_engine.host = self

    def attach_inference_engine(self, inference_engine, **kwargs):
        """Attach an inference engine

        Helper function to attach an inference engine.

        :param inference: a CoopIHC inference engine
        :type inference: :doc:mod:`Inference Engine<coopihc/inference>`

        :meta private:
        """
        if inference_engine is None:
            inference_engine = BaseInferenceEngine()
        else:
            inference_engine = inference_engine

        if type(inference_engine).__name__ == "type":
            self.inference_engine = inference_engine(**kwargs)
        else:
            self.inference_engine = inference_engine
            if kwargs != {}:
                raise AttributeError(
                    "Can't input an instantiated inference engine and associated keyword arguments. Either pass the inference engine class, or fully instantiate that policy before passing it."
                )

        self.inference_engine.host = self

    def _base_reset(self, all=True, dic=None):
        """Reset function called by the Bundle.

        This method is called by the bundle to reset the agent. It defines a bunch of actions that should be performed upon each reset. It namely calls the reset method that can be modified by the end-user of the library.



        :param all: which components to reset, defaults to True
        :type all: bool, optional
        :param dic: reset dictionary, defaults to None.
        :type dic: [type], optional

        :meta private:
        """

        if all:
            self.policy.reset()
            self.inference_engine.reset()
            self.observation_engine.reset()

        if not dic:
            self.state.reset()
            # self.reset(dic=dic)   # Check but this should not be needed
            self.reset()

            return

        # self.reset(dic=dic)   # Check but this should not be needed
        self.reset()  # Reset all states before, just in case the reset dic does not specify a reset value for each substate.
        for key in list(self.state.keys()):
            value = dic.get(key)
            if isinstance(value, StateElement):
                value = value["values"]
            if value is not None:
                self.state[key]["values"] = value

    def reset(self):
        """Initialize the agent before each new game.

        Specify how the components of the agent will be reset. By default, the agent will call the reset method of all 4 components (policy, inference engine, observation engine, state). You can specify some added behavior here e.g. if you want to have a fixed value for the state at the beggining of each game (default behavior is random), you can speficy that here:

        .. code-block:: python

            # Sets the value of state 'x' to 0
            self.state["x"]["values"] = [
                numpy.array([0])
            ]

        :meta public:
        """
        pass

    def finit(self):
        """Finish initializing.

        Method that specifies what happens when initializing the agent for the very first time (similar to __init__), but after the bundle has been initialized already. This allows to finish initializing (finit) the agent when information from another component is required to do so e.g. an assistant which requires the list of possible targets from the task.

        :meta public:
        """

        pass

    def _take_action(self):
        """Take action.

        What the agent should do when the Bundle expects it to take an action.

        :return: return action and reward
        :rtype: tuple(coopihc.space.State.State, float)

        :meta private:
        """
        return self.policy.sample()

    def _agent_step(self, infer=True):
        """Play an agent's turn.

        Observe the game state via the observation engine, update the internal state via the inference engine, collect rewards for both processes and return them to the caller (the bundle).

        :param infer: whether inference should be performed, defaults to True
        :type infer: bool, optional
        :return: observation and inference rewards
        :rtype: tuple(float, float)

        :meta private:
        """
        # agent observes the state
        agent_observation, agent_obs_reward = self._observe(self.bundle.game_state)

        # Pass observation to InferenceEngine Buffer
        self.inference_engine.add_observation(agent_observation)
        # Infer the new user state
        if infer:

            agent_state, agent_infer_reward = self.inference_engine.infer()
            # Broadcast new agent_state
            self.state.update(agent_state)

            # Update agent observation
            if self.role == "user":
                if self.inference_engine.buffer[-1].get("user_state") is not None:
                    self.inference_engine.buffer[-1]["user_state"].update(agent_state)
            elif self.role == "assistant":
                if self.inference_engine.buffer[-1].get("assistant_state") is not None:
                    self.inference_engine.buffer[-1]["assistant_state"].update(
                        agent_state
                    )
        else:
            agent_infer_reward = 0
        return agent_obs_reward, agent_infer_reward

    def _observe(self, game_state):
        """Observe gamestate.

        Observe the gamestate by calling to the observation engine.

        :param game_state: current game state
        :type game_state: coopihc.space.State.State
        :return: return observation and associated reward
        :rtype: tuple(coopihc.space.State.State, float)

        :meta private:
        """
        observation, reward = self.observation_engine.observe(game_state)
        return observation, reward

    def render(self, *args, **kwargs):
        """Renders the agent.

        Render can be redefined but should keep the same signature. Currently supports text and plot modes.

        :param args: (list) list of axes used in the bundle render, in order: axtask, axuser, axassistant
        :param mode: (str) currently supports either 'plot' or 'text'. Both modes can be combined by having both modes in the same string e.g. 'plottext' or 'plotext'.

        :meta public:
        """
        mode = kwargs.get("mode")
        if mode is None:
            mode = "text"

        if "plot" in mode:
            axtask, axuser, axassistant = args[:3]
            if self.ax is not None:
                pass
            else:
                if self.role == "user":
                    self.ax = axuser
                else:
                    self.ax = axassistant
                self.ax.axis("off")
                self.ax.set_title(type(self).__name__ + " State")
        if "text" in mode:
            print(type(self).__name__ + " State")
