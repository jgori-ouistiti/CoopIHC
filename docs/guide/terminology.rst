.. terminology:

Terminology
===============

Model:

* user: The human user that is taking part in an interaction.
* user: an artificial model of a user
* assistant: an intelligent tool or interface
* agents: some entity that can produce observations, perform inferences and take actions. In practice this is used to qualify either the user or assistant.
* interaction task, task: a task defined by a given state, to be solved by the user with the assistant's aid. A task is solved when it reaches a certain goal state.
* interaction game, game: the game formed by combining a task, an user, and an assistant


Module:

* modeler: that's you
* bundle: the component that turns a task, an user and an assistant into an interaction game
* user model: the description of the policy of the user, expressed as a likelihood and a sampling rule
* observation engine: the agent's component that produces observations from the game state
* inference engine: the agent's component that is responsible for modifying its internal state
* base agent: the most basic agent, with a random policy, an internal state that is not updated and an observation engine that allows perfect observations of the game and own internal states.
* run: all the steps performed on a bundle between its reset and its termination. When bundle is an environment, runs and episodes coïncide.
* sampling, sample: selecting an action from the possible action set.

Gym:

* step: run a timestep of the environment's dynamics
* environment, env: an object with arbitrary dynamics, with a standard API compatible with most off-the-shelf RL algorithms.
* render: Produce a display of the environment
* reset: Force the state of the environment to an initial state.

Bundle:

* step: run a timestep of the bundle's dynamics
* agent step: observe and infer, return observation and inference rewards
* turn: number of agent steps since last round
* round: number of bundle steps since reset
* env (train wrapper): a bundle that has been wrapped ``Train`` . Equivalent to a gym environment
