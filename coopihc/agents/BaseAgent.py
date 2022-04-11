from coopihc.base.State import State
from coopihc.base.StateElement import StateElement
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.policy.BasePolicy import BasePolicy
from coopihc.observation.RuleObservationEngine import RuleObservationEngine
from coopihc.observation.utils import base_user_engine_specification
from coopihc.observation.utils import base_assistant_engine_specification
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine

import numpy
import copy


class BaseAgent:
    """A *Coopihc* agent.


    Instantiate or subclass this class to define an agent that is compatible with *CoopIHC*.

    An agent has 4 components:

        * An internal state
        * An observation engine, that produces observations of the task and the other agent
        * An inference engine, that uses this observation to make the internal state transition towards a new state
        * A policy, which, based on the agent's internal state and its observation, picks an action.



    By default, this class will be initialized with an empty internal :py:class:`State<coopihc.base.State>`, a random binary :py:class:`BasePolicy<coopihc.policy.BasePolicy>`, a :py:class:`RuleObservationEngine<coopihc.observation.RuleObservationEngine>` that sees everything except the other agent's internal state, and a :py:class:`BaseInference<coopihc.observation.BaseInference>` engine which does not update the state.



    The API methods that users of this class can redefine are:

        + ``finit``: a second round of initialization once a bundle has been formed -- useful because at that point the agent has a reference to the other agent and task.
        + ``reset``: to specify how to initialize the agent's state at the end of each game. Policies, inference engines, and observation engines handle their own resets methods.
        + ``render``: specifies what to display.


    Some things to know:

    * The agent can be used to produce observations, inferences and actions outside of any Bundle. See methods ``observe(), infer(), take_action()``.

    * You can override some components, e.g. to override the existing policy of an agent named ``MyNewUser`` with some other policy, you can do the following

    .. code-block:: python

        changed_policy_user = MyNewUser(override_agent_policy = (some_other_policy, other_policy_kwargs))


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


    :return: A *CoopIHC* and :py:class:`Bundle<coopihc.bundle>`-compatible agent
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
        **kwargs,
    ):

        # Bundles stuff
        self._bundle = None
        self._bundle_memory = None
        self.ax = None
        self._parameters = {}

        # Set role of agent
        if role not in ["user", "assistant"]:
            raise ValueError(
                "First argument 'role' should be either 'user' or 'assistant'"
            )
        else:
            self.role = role

        # Define policy
        self._attach_policy(agent_policy, **policy_kwargs)

        # Init state
        if agent_state is None:
            self._state = State(**state_kwargs)
        else:
            self._state = agent_state

        # Define observation engine
        self._attach_observation_engine(
            agent_observation_engine, **observation_engine_kwargs
        )

        # Define inference engine
        self._attach_inference_engine(agent_inference_engine, **inference_engine_kwargs)

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

        :meta private:
        """
        # Override agent policy
        agent_policy, agent_policy_kwargs = init_kwargs.get(
            "override_policy", (None, None)
        )
        if agent_policy is not None:
            self._attach_policy(agent_policy, **agent_policy_kwargs)

        # Override agent state
        agent_state = init_kwargs.get("override_state", None)
        if agent_state is not None:
            self._state = agent_state

        # Override agent observation engine
        agent_obseng, agent_obseng_kwargs = init_kwargs.get(
            "override_observation_engine", (None, None)
        )
        if agent_obseng is not None:
            self._attach_observation_engine(agent_obseng, **agent_obseng_kwargs)

        # Override agent inference engine
        agent_infeng, agent_infeng_kwargs = init_kwargs.get(
            "override_inference_engine", (None, None)
        )
        if agent_infeng is not None:
            self._attach_inference_engine(agent_infeng, **agent_infeng_kwargs)

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
    def parameters(self):
        if self.bundle:
            return self.bundle.parameters
        return self._parameters

    @property
    def bundle(self):
        return self._bundle

    @bundle.setter
    def bundle(self, value):
        if type(value).__name__ == "Simulator":
            self.bundle_memory = copy.copy(self._bundle)
        self._bundle = value

    @property
    def bundle_memory(self):
        return self._bundle_memory

    @bundle_memory.setter
    def bundle_memory(self, value):
        if type(value).__name__ == "Simulator":
            return
        self._bundle_memory = value

    def _simulator_close(self):
        self._bundle = self._bundle_memory

    @property
    def policy(self):
        """Agent policy"""
        return self._policy

    @policy.setter
    def policy(self, value):
        self._attach_policy(value)

    @property
    def inference_engine(self):
        """Agent inference engine"""
        return self._inference_engine

    @inference_engine.setter
    def inference_engine(self, value):
        self._attach_inference_engine(value)

    @property
    def observation_engine(self):
        """Agent observation engine"""
        return self._observation_engine

    @observation_engine.setter
    def observation_engine(self, value):
        self._attach_observation_engine

    @property
    def state(self):
        """Agent internal state"""
        return self._state

    @property
    def observation(self):
        """Last agent observation"""
        return (
            self.inference_engine.buffer[-1]
            if self.inference_engine.buffer is not None
            else None
        )

    @property
    def action(self):
        """Last agent action"""
        return self.policy.action

    @action.setter
    def action(self, item):
        self.policy.action = item

    @property
    def user(self):
        """Connected user"""
        if self.role == "user":
            return self
        else:
            try:
                return self.bundle.user
            except AttributeError:  # No bundle known
                raise AttributeError(
                    f"Agent{self.__class__.__name__} has not been connected to a user yet."
                )

    @property
    def assistant(self):
        """Connected assistant"""
        if self.role == "assistant":
            return self
        else:
            try:
                return self.bundle.assistant
            except AttributeError:  # No bundle known
                raise AttributeError(
                    f"Agent{self.__class__.__name__} has not been connected to a assistant yet."
                )

    @property
    def task(self):
        """Connected task"""
        try:
            return self.bundle.task
        except AttributeError:  # No bundle known
            raise AttributeError(
                f"Agent{self.__class__.__name__} has not been connected to a task yet."
            )

    # def __getattr__(self, value):
    #     try:
    #         return self.parameters.__getitem__(value)
    #     except:
    #         raise AttributeError(
    #             f"{self.__class__.__name__} object has no attribute {value}"
    #         )

    def _attach_policy(self, policy, **kwargs):
        """Attach a policy

        Helper function to attach a policy.

        :param policy: a CoopIHC policy
        :type policy: :doc:mod:`Policy<coopihc/policy>`

        :meta private:
        """
        if policy is None:
            policy = BasePolicy

        if type(policy).__name__ == "type":
            self._policy = policy(**kwargs)
        else:
            self._policy = policy
            if kwargs != {}:
                raise AttributeError(
                    "Can't input an instantiated policy and associated keyword arguments. Either pass the policy class, or fully instantiate that policy before passing it."
                )

        self._policy.host = self

    def _attach_observation_engine(self, observation_engine, **kwargs):
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
            self._observation_engine = observation_engine(**kwargs)
        else:
            self._observation_engine = observation_engine
            if kwargs != {}:
                raise AttributeError(
                    "Can't input an instantiated observation engine and associated keyword arguments. Either pass the observation engine class, or fully instantiate that policy before passing it."
                )

        self._observation_engine.host = self

    def _attach_inference_engine(self, inference_engine, **kwargs):
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
            self._inference_engine = inference_engine(**kwargs)
        else:
            self._inference_engine = inference_engine
            if kwargs != {}:
                raise AttributeError(
                    "Can't input an instantiated inference engine and associated keyword arguments. Either pass the inference engine class, or fully instantiate that policy before passing it."
                )

        self._inference_engine.host = self

    def _base_reset(self, all=True, dic=None, random=True):
        """Reset function called by the Bundle.

        This method is called by the bundle to reset the agent. It defines a bunch of actions that should be performed upon each reset. It namely calls the reset method that can be modified by the end-user of the library.



        :param all: which components to reset, defaults to True
        :type all: bool, optional
        :param dic: reset dictionary, defaults to None.
        :type dic: [type], optional

        :meta private:
        """
        if all:
            self.policy.reset(random=random)
            self.inference_engine.reset(random=random)
            self.observation_engine.reset(random=random)

        if not dic:
            if random:
                self.state.reset()
            self.reset()

            return

        # forced reset with dic
        for key in list(self.state.keys()):
            value = dic.get(key)
            if isinstance(value, StateElement):
                self.state[key] = value
                continue
            elif isinstance(value, numpy.ndarray):
                self.state[key][...] = value

            elif value is None:
                continue
            else:
                try:
                    self.state[key][
                        ...
                    ] = value  # Give StateElement's preprocessvalues method a chance
                except:
                    raise NotImplementedError

    def reset(self):
        """reset the agent --- Override this

        Override this method to specify how the components of the agent will be reset. By default, the agent will already call the reset method of all 4 components (policy, inference engine, observation engine, state). You can specify some added behavior here e.g. if you want each game to begin with a specific state value, you can specify that here. For example:

        .. code-block:: python

            # Sets the value of state 'x' to 0
            def reset(self):
                self.state["x"][...] = 123

        :meta public:
        """
        pass

    def reset_all(self, dic=None, random=True):
        """reset the agent and all its components

        In addition to running the agent's ``reset()``, ``reset_all()`` also calls state, observation engine, inference engine and policies' ``reset()`` method.

        :param dic: reset_dictionnary, defaults to None. See the ``reset()`` method in `py:class:Bundle<coopihc.bundle.Bundle>` for more information.
        :type dic: dictionary, optional
        :param random: whether states should be randomly reset, defaults to True. See the ``reset()`` method in `py:class:Bundle<coopihc.bundle.Bundle>` for more information.
        :type random: bool, optional

        :meta public:
        """
        self._base_reset(all=True, dic=dic, random=random)

    def finit(self):
        """Finish initializing.

        Method that specifies what happens when initializing the agent for the very first time (similar to __init__), but after a bundle has been initialized already. This allows to finish initializing (finit) the agent when information from another component is required to do so.

        :meta public:
        """

        pass

    def take_action(
        self,
        agent_observation=None,
        agent_state=None,
        increment_turn=True,
    ):
        """Select an action

        Select an action based on agent_observation and agent_state, by querying the agent's policy. If either of these arguments is not provided, then the argument is deduced from the agent's internals.

        :param agent_observation: last agent observation, defaults to None. If None, gets the observation from the inference engine's buffer.
        :type agent_observation: :py:class:State<coopihc.base.State>, optional
        :param agent_state: current value of the agent's internal state, defaults to None. If None, gets the state from itself.
        :type agent_state: :py:class:State<coopihc.base.State>, optional
        :param increment_turn: whether to update bundle's turn and round
        :type increment_turn: bool, optional

        :meta public:
        """

        try:
            if increment_turn:
                self.bundle.turn_number = (self.bundle.turn_number + 1) % 4
                if self.bundle.turn_number == 0:
                    self.bundle.round_number += 1
        except AttributeError:  # Catch case where agent not linked to a bundle
            if self.bundle is None:
                pass
            else:  # Re-raise exception
                self.bundle.turn_number = (self.bundle.turn_number + 1) % 4

        return self.policy._base_sample(
            agent_observation=agent_observation, agent_state=agent_state
        )

    def observe(
        self,
        game_state=None,
        affect_bundle=True,
        game_info={},
        task_state={},
        user_state={},
        assistant_state={},
        user_action={},
        assistant_action={},
    ):
        """produce an observation

        Produce an observation based on state information, by querying the agent's observation engine. By default, the agent will find the appropriate states to observe. To bypass this behavior, you can provide state information. When doing so, either provide the full game state, or provide the needed individual states.
        The affect_bundle flag determines whether or not the observation produces like this becomes the agent's last observation.

        :param game_state: the full game state as defined in the *CoopIHC* interaction model, defaults to None.
        :type game_state: `:py:class:State<coopihc.base.State>`, optional
        :param affect_bundle: whether or not the observation is stored and becomes the agent's last observation, defaults to True.
        :type affect_bundle: bool, optional
        :param game_info: game_info substate, see the *CoopIHC* interaction model, defaults to {}.
        :type game_info: `:py:class:State<coopihc.base.State>`, optional
        :param task_state: task_state substate, see the *CoopIHC* interaction model, defaults to {}
        :type task_state: `:py:class:State<coopihc.base.State>`, optional
        :param user_state: user_state substate, see the *CoopIHC* interaction model, defaults to {}
        :type user_state: `:py:class:State<coopihc.base.State>`, optional
        :param assistant_state: assistant_state substate, see the *CoopIHC* interaction model, defaults to {}
        :type assistant_state: `:py:class:State<coopihc.base.State>`, optional
        :param user_action: user_action substate, see the *CoopIHC* interaction model, defaults to {}
        :type user_action: `:py:class:State<coopihc.base.State>`, optional
        :param assistant_action: assistant_action substate, see the *CoopIHC* interaction model, defaults to {}
        :type assistant_action: `:py:class:State<coopihc.base.State>`, optional

        :meta public:
        """
        if (
            bool(game_info)
            or bool(task_state)
            or bool(user_state)
            or bool(assistant_state)
            or bool(user_action)
            or bool(assistant_action)
        ):
            observation, reward = self.observation_engine.observe_from_substates(
                game_info=game_info,
                task_state=task_state,
                user_state=user_state,
                assistant_state=assistant_state,
                user_action=user_action,
                assistant_action=assistant_action,
            )
        else:
            observation, reward = self.observation_engine.observe(game_state=game_state)
        if affect_bundle:
            self.inference_engine.add_observation(observation)
        return observation, reward

    def infer(self, agent_observation=None, affect_bundle=True):
        """infer the agent's internal state

        Infer the new agent state from the agent's observation. By default, the agent will select the agent's last observation. To bypass this behavior, you can provide a given agent_observation.
        The affect_bundle flag determines whether or not the agent's internal state is actually updated.


        :param agent_observation: last agent observation, defaults to None. If None, gets the observation from the inference engine's buffer.
        :type agent_observation: :py:class:State<coopihc.base.State>, optional
        :param affect_bundle: whether or not the agent's state is updated with the new inferred state, defaults to True.
        :type affect_bundle: bool, optional

        :meta public:
        """
        agent_state, agent_infer_reward = self.inference_engine.infer(
            agent_observation=agent_observation
        )
        if affect_bundle:
            self.state.update(agent_state)
        return agent_state, agent_infer_reward

    def _agent_step(self, infer=True):
        """Play an agent's turn.

        Observe the game state via the observation engine, update the internal state via the inference engine, collect rewards for both processes and return them to the caller (the bundle).

        :param infer: whether inference should be performed, defaults to True
        :type infer: bool, optional
        :return: observation and inference rewards
        :rtype: tuple(float, float)

        :meta private:
        """
        # agent_observation, agent_obs_reward = self.observe(self.bundle.game_state)
        agent_observation, agent_obs_reward = self.observe()

        if infer:
            agent_state, agent_infer_reward = self.infer()
        else:
            agent_infer_reward = 0
        return agent_obs_reward, agent_infer_reward

    def prepare_action(
        self,
        affect_bundle=True,
        game_state=None,
        agent_observation=None,
        increment_turn=True,
        **kwargs,
    ):
        if self.bundle is not None:
            if self.bundle.turn_number != 0 and self.role == "user":
                raise RuntimeError(
                    f"You are preparing User {self.__class__.__name__} to take an action, but the Bundle is at turn {self.bundle.turn_number} (should be 0) "
                )
            if self.bundle.turn_number != 2 and self.role == "assistant":
                raise RuntimeError(
                    f"You are preparing Assistant {self.__class__.__name__} to take an action, but the Bundle is at turn {self.bundle.turn_number} (should be 2) "
                )

            if increment_turn:
                self.bundle.turn_number = (self.bundle.turn_number + 1) % 4

        if agent_observation is None:
            _agent_observation, agent_obs_reward = self.observe(
                affect_bundle=affect_bundle, game_state=game_state, **kwargs
            )
            if agent_observation is None:
                agent_observation = _agent_observation
            agent_state, agent_infer_reward = self.infer(
                agent_observation=agent_observation, affect_bundle=affect_bundle
            )
        return agent_obs_reward + agent_infer_reward

    def render(self, mode="text", ax_user=None, ax_assistant=None, ax_task=None):
        """render the agent

        Displays agent information on the passed axes.

        :param mode: display mode, defaults to "text". Also supports "plot".
        :type mode: str, optional
        :param ax_user: user axis, defaults to None
        :type ax_user: Matploblib axis, optional
        :param ax_assistant: assistant axis, defaults to None
        :type ax_assistant: Matploblib axis, optional
        :param ax_task: task axis, defaults to None
        :type ax_task: Matploblib axis, optional
        """

        if "plot" in mode:
            if self.ax is not None:
                pass
            else:
                if self.role == "user":
                    self.ax = ax_user
                else:
                    self.ax = ax_assistant
                self.ax.axis("off")
                self.ax.set_title(type(self).__name__ + " State")
        if "text" in mode:
            print(type(self).__name__ + " State")
