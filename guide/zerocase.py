from coopihc.interactiontask import InteractionTask
from coopihc.base import StateElement
import gym
import numpy


class ExampleTask(InteractionTask):
    """ExampleTask with two agents. A cursor is moved from a given position through the multiplication of the actions of a simulated user and an assistant."""

    def __init__(self, gridsize=10, number_of_targets=2):
        # Call super().__init__() beofre anything else, which initializes some useful attributes, including a State (self.state) for the task
        super().__init__()

        # Store invariant parameters as attributes
        self.gridsize = gridsize
        self.number_of_targets = number_of_targets

        # Store other parameters in State
        self.state["Position"] = StateElement(
            values=None, spaces=[gym.spaces.Discrete(gridsize)], possible_values=None
        )
        self.state["Targets"] = StateElement(
            values=None,
            spaces=[
                gym.spaces.Discrete(gridsize) for i in range(self.number_of_targets)
            ],
            possible_values=None,
        )

    def reset(self, dic=None):
        # Pick random targets, and a starting position (all different)
        locations = sorted(
            numpy.random.choice(
                list(range(self.gridsize)),
                size=self.number_of_targets + 2,
                replace=False,
            )
        )
        targets, position, goal = locations[:-2], locations[-2], locations[-1]
        self.state["Targets"]["values"] = targets
        self.state["Position"]["values"] = position
        self.goal = goal
        self.grid = [" " for i in range(self.gridsize)]
        for i in targets:
            self.grid[i] = "T"
        super().reset(dic)

    def user_step(self, *args, **kwargs):
        return super().user_step()

    def assistant_step(self, *args, **kwargs):
        # Call super method before anything else
        super().assistant_step()

        is_done = False

        # Look up needed inputs in the game state, and use the special 'human_values' key to convert to human readable values
        assistant_action = self.bundle.game_state["assistant_action"]["action"][
            "human_values"
        ][0]
        user_action = self.bundle.game_state["user_action"]["action"]["human_values"][0]
        position = self.state["Position"]["human_values"][0]

        # Apply modulation, with rounding and clipping.
        self.state["Position"]["values"] = [
            int(
                numpy.clip(
                    numpy.round(position + user_action * assistant_action, decimals=0),
                    0,
                    self.gridsize - 1,
                )
            )
        ]

        # Check if the Goal is attained
        if self.state["Position"]["human_values"][0] == self.goal:
            is_done = True
        return self.state, -1 / 2, is_done, {}

    def render(self, *args, mode="text"):
        """Render the task.

        Plot or print the grid, with cursor and target positions.

        :param ax:
        :param args:
        :param mode: 'text' or 'plot'

        .. warning::

            revisit the signature of this function

        :meta public:
        """
        self.grid[self.goal] = "G"
        if "text" in mode:
            tmp = self.grid.copy()
            tmp[int(self.state["Position"]["human_values"][0])] = "P"
            _str = "|"
            for t in tmp:
                _str += t + "|"

            print("\n")
            print("Turn number {:f}".format(self.turn))
            print(_str)

            targets = sorted(self.state["Targets"]["human_values"])
            print("Targets:")
            print(targets)
            print("\n")
        else:
            raise NotImplementedError


task = ExampleTask(gridsize=15, number_of_targets=3)
print(task.state)
task.reset()
print(task.state)

from coopihc.bundle import _DevelopTask

bundle = _DevelopTask(task)
bundle.render("text")
bundle.step([-1, 1])
bundle.render("text")


#
#
# if _str == 'basic-plot':
#     task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
#     user = CarefulPointer()
#     assistant = ConstantCDGain(1)
#
#     bundle = PlayNone(task, user, assistant)
#     game_state = bundle.reset()
#     bundle.render('plotext')
#
# if _str == 'basic-PlayNone' or _str == 'all':
#     task = SimplePointingTask(gridsize = 31, number_of_targets = 8)
#     binary_user = CarefulPointer()
#     action_space = [gym.spaces.Discrete(1)]
#     action_set = [[1]]
#     agent_policy = Policy(action_space, action_set = action_set)
#
#
#     unitcdgain = BaseAgent( 'assistant',
#                             policy = agent_policy,
#                             observation_engine = None,
#                             inference_engine = None
#                             )
#
#     bundle = PlayNone(task, binary_user, unitcdgain)
#     game_state = bundle.reset()
#     bundle.render('plotext')
#     while True:
#         sum_rewards, is_done, rewards = bundle.step()
#         bundle.render('plotext')
#         if is_done:
#             bundle.close()
#             break
