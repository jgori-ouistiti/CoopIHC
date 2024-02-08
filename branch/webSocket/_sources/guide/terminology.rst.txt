.. terminology:

Terminology
===============

Model
-----------

* user: The agent that is taking part in an interaction with a certain goal. It may be a real human user, or a synthetic user model.
* assistant: an intelligent tool or interface that assists the user.
* agents: some entity that can produce observations, perform inferences and take actions. Refers in practice to both users and assistants.
* interaction task, task: a task defined by a given state, to be solved by the user with the assistant's aid. A task is solved when it reaches a certain goal state.
* interaction game, game: the two-agent game formed by combining a task, a user, and an assistant.


Module
-------

* bundle: the component that turns a task, a user and an assistant into an interaction game
* observation engine: the agent's component that produces observations from the game state
* inference engine: the agent's component that is responsible for modifying its internal state, based on the previous state and the available observations.
* sampling, sample: selecting an action from the possible action set. Usually this is not random sampling, but based on some policy.
* round: Each sequence of joint decisions and task updates make a round (just like in a boardgame, a round is when everyone has played its turn)
* turn: There are 4 turns per round (observation and inference, taking action, both for the user and the assistant).
* step: round
* reset: Force the state of the environment to an initial (potentially random) state.


