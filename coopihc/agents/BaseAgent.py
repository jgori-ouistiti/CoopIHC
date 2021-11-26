from abc import ABC
from coopihc.space import State, StateElement
from coopihc.policy import BasePolicy
from coopihc.observation import (
    RuleObservationEngine,
    base_user_engine_specification,
    base_assistant_engine_specification,
)
from coopihc.inference import BaseInferenceEngine


class BaseAgent(ABC):
    """Instantiate or subclass this class to define an agent that can be used in a Bundle.

    By default, this class will be initialized with an empty internal state, a BasePolicy with a single 'None' action, a RuleObservationEngine with a BaseUser of BaseAssistant profile, and a BaseInference engine.

    You can override some components, e.g. to override the existing policy of an agent, do:
        changed_policy_user = MyNewUser(agent_policy = some_other_policy)
    In that case, MyNewUser class which had been defined with some specific policy, has had its policy overriden by 'some_other_policy'; all other components remain equal.


    The main API methods that users of this class should redefine are:

        + finit
        + reset
        + render


    :param str role: 'user' or 'assistant'
    :param type **kwargs: keyword values ( each agent_X key expects a valid X object, and X_kwargs expects a valid dictionnary of keyword arguments for X)

        + agent_policy
        + agent_inference_engine
        + agent_observation_engine
        + agent_state
        + policy_kwargs
        + inference_engine_kwargs
        + observation_engine_kwargs
        + state_kwargs


    :return: A Bundle-compatible agent
    :rtype: BaseAgent

    """

    def __init__(self, role, **kwargs):
        component_dic, remaining_kwargs = self._allow_override(
            **kwargs
        )  # This line can probably be removed

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
        self.attach_policy(component_dic["policy"], **kwargs.pop("policy_kwargs", {}))
        # self._attach_policy(policy, **policy_kwargs)

        # Init state
        if component_dic["state"] is None:
            self.state = State(**kwargs.pop("state_kwargs", {}))
        else:
            self.state = component_dic["state"]

        # Define observation engine
        self.attach_observation_engine(
            component_dic["observation_engine"],
            **kwargs.pop("observation_engine_kwargs", {})
        )

        # Define inference engine
        self.attach_inference_engine(
            component_dic["inference_engine"],
            **kwargs.pop("inference_engine_kwargs", {})
        )

    def _allow_override(self, **kwargs):
        ## Seems to me that the commented out part is useless. I forgot why I did it in the first place.

        # oap = kwargs.pop("override_agent_policy", None)
        # if oap is not None:
        #     agent_policy = oap
        # else:
        agent_policy = kwargs.pop("agent_policy", None)

        # oaoe = kwargs.pop("override_agent_observation_engine", None)
        # if oaoe is not None:
        #     agent_observation_engine = oaoe
        # else:
        agent_observation_engine = kwargs.pop("agent_observation_engine", None)

        # oaie = kwargs.pop("override_agent_inference_engine", None)
        # if oaie is not None:
        #     agent_inference_engine = oaie
        # else:
        agent_inference_engine = kwargs.pop("agent_inference_engine", None)

        # oas = kwargs.pop("override_agent_state", None)
        # if oas is not None:
        #     agent_state = oas
        # else:
        agent_state = kwargs.pop("agent_state", None)

        return {
            "state": agent_state,
            "policy": agent_policy,
            "observation_engine": agent_observation_engine,
            "inference_engine": agent_inference_engine,
        }, kwargs

    def _content__(self):
        return {
            "Name": self.__class__.__name__,
            "State": self.state._content__(),
            "Observation Engine": self.observation_engine._content__(),
            "Inference Engine": self.inference_engine._content__(),
            "Policy": self.policy._content__(),
        }

    @property
    def observation(self):
        """Returns the latest observation"""
        return (
            self.inference_engine.buffer[-1]
            if self.inference_engine.buffer is not None
            else None
        )

    @property
    def action(self):
        """Returns the latest selected action"""
        return self.policy.action_state["action"]

    def attach_policy(self, policy, **kwargs):
        if policy is None:
            self.policy = BasePolicy()
        else:
            self.policy = policy
        self.policy.host = self

    def attach_observation_engine(self, observation_engine, **kwargs):
        if observation_engine is None:
            if self.role == "user":
                self.observation_engine = RuleObservationEngine(
                    base_user_engine_specification
                )
            elif self.role == "assistant":
                self.observation_engine = RuleObservationEngine(
                    base_assistant_engine_specification
                )
            else:
                raise NotImplementedError
        else:
            self.observation_engine = observation_engine
        self.observation_engine.host = self

    def attach_inference_engine(self, inference_engine, **kwargs):
        if inference_engine is None:
            self.inference_engine = BaseInferenceEngine()
        else:
            self.inference_engine = inference_engine
        self.inference_engine.host = self

    def _base_reset(self, all=True, dic=None):

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
        self.reset()
        for key in list(self.state.keys()):
            value = dic.get(key)
            if isinstance(value, StateElement):
                value = value["values"]
            if value is not None:
                self.state[key]["values"] = value

    def reset(self):
        """What should happen when initializing the agent.

        Bundles will reset all engines, states and policies and handles forced resets (by the reset dictionnary mechanism), so you don't need to take care of that here.

        """
        pass

    def finit(self):
        """finit is called by bundle during its initialization. This gives the possibility to finish initializing (finit) the agent when information from another component is required e.g. an assistant which requires the list of possible targets from the task."""
        pass

    def _take_action(self):
        return self.policy.sample()

    def _agent_step(self, infer=True):
        """Play one agent's turn: Observe the game state via the observation engine, update the internal state via the inference engine, collect rewards for both processes and pass them to the caller (usually the bundle).

        :return: agent_obs_reward; agent_infer_reward: agent_obs_reward (float) reward associated with observing. agent_infer_reward (float) reward associated with inferring

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
        observation, reward = self.observation_engine.observe(game_state)
        return observation, reward

    def render(self, *args, **kwargs):
        """Renders the agent part of the bundle. Render can be redefined but should keep the same signature. Currently supports text and plot modes.

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
