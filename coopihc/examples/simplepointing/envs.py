import numpy
import matplotlib.pyplot as plt
from collections import OrderedDict
import coopihc
from coopihc.interactiontask.InteractionTask import InteractionTask
from coopihc.interactiontask.PipeTaskWrapper import PipeTaskWrapper

PipeTaskWrapper
from coopihc.base.Space import Space
from coopihc.base.StateElement import StateElement
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.helpers import flatten
from coopihc.helpers import sort_two_lists
import functools


class SimplePointingTask(InteractionTask):
    """A 1D pointing task.

    A 1D grid of size 'Gridsize'. The cursor is at a certain 'position' and there are several potential 'targets' on the grid. The user action is modulated by the assistant.

    :param gridsize: (int) Size of the grid
    :param number_of_targets: (int) Number of targets on the grid

    :meta public:
    """

    @property
    def user_action(self):
        return super().user_action[0]

    @property
    def assistant_action(self):
        return super().assistant_action[0]

    def __init__(self, gridsize=31, number_of_targets=10, mode="gain"):
        super().__init__()
        self.gridsize = gridsize
        self.number_of_targets = number_of_targets
        self.mode = mode
        self.dim = 1

        self.state["position"] = discrete_array_element(
            low=0, high=gridsize - 1, out_of_bounds_mode="clip"
        )

        self.state["targets"] = discrete_array_element(
            low=0, high=gridsize, shape=(number_of_targets,)
        )

    def reset(self, dic=None):
        """Reset the task.

        Reset the grid used for rendering, define new targets, select a starting position

        :param args: reset to the given state

        :meta public:
        """
        self.grid = [" " for i in range(self.gridsize)]
        targets = sorted(
            numpy.random.choice(
                list(range(self.gridsize)), size=self.number_of_targets, replace=False
            )
        )
        for i in targets:
            self.grid[i] = "T"
        # Define starting position
        copy = list(range(len(self.grid)))
        for i in targets:
            copy.remove(i)
        position = int(numpy.random.choice(copy))
        self.state["position"][...] = position
        self.state["targets"][...] = targets

    def on_user_action(self, *args, user_action=None, **kwargs):
        """Do nothing, increment turns, return half a timestep

        :meta public:
        """
        is_done = False
        if (
            # self.user_action == 0
            # and self.state["position"] == self.bundle.user.state["goal"]
            self.state["position"]
            == self.bundle.user.state["goal"]
        ):
            is_done = True
        return self.state, -1, is_done

    def on_assistant_action(self, *args, assistant_action=None, **kwargs):
        """Modulate the user's action.

        Multiply the user action with the assistant action.
        Update the position and grids.

        :param assistant_action: (list)

        :return: new state (OrderedDict), half a time step, is_done (True/False)

        :meta public:
        """
        is_done = False

        # Stopping condition if too many turns
        if int(self.round_number) >= 50:
            return self.state, 0, True

        if self.mode == "position":
            self.state["position"][...] = self.assistant_action
        elif self.mode == "gain":

            position = self.state["position"]

            self.state["position"][...] = numpy.round(
                position + self.user_action * self.assistant_action
            )

        return self.state, 0, False

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
        goal = self.bundle.game_state["user_state"]["goal"].squeeze().tolist()
        self.grid[goal] = "G"
        if "text" in mode:
            tmp = self.grid.copy()
            tmp[int(self.state["position"].squeeze().tolist())] = "P"
            _str = "|"
            for t in tmp:
                _str += t + "|"

            print(_str)

            targets = sorted(self.state["targets"])
            print("Targets:")
            print([t.squeeze().tolist() for t in targets])
            print("\n")
        if "plot" in mode:
            axtask, axuser, axassistant = args[:3]
            if self.ax is not None:
                self.update_position()
            else:
                self.ax = axtask
                self.init_grid()
                self.ax.set_aspect("equal")
        if not ("plot" in mode or "text" in mode):
            raise NotImplementedError

    def set_box(
        self,
        ax,
        pos,
        draw="k",
        fill=None,
        symbol=None,
        symbol_color=None,
        shortcut=None,
    ):
        """For Render.

        Draws box on the given axis ax at the given position pos, with a given style.

        :meta public:
        """
        if shortcut == "void":
            draw = "k"
            fill = "#aaaaaa"
            symbol = None
        elif shortcut == "target":
            draw = "#96006c"
            fill = "#913979"
            symbol = "$T$"
            symbol_color = "k"
        elif shortcut == "goal":
            draw = "#009c08"
            fill = "#349439"
            symbol = "$G$"
            symbol_color = "k"
        elif shortcut == "position":
            draw = "#00189c"
            fill = "#6573bf"
            symbol = "$P$"
            symbol_color = "k"

        BOX_SIZE = 1
        BOX_HW = BOX_SIZE / 2

        _x = [pos - BOX_HW, pos + BOX_HW, pos + BOX_HW, pos - BOX_HW]
        _y = [-BOX_HW, -BOX_HW, BOX_HW, BOX_HW]
        x_cycle = _x + [_x[0]]
        y_cycle = _y + [_y[0]]
        if fill is not None:
            fill = ax.fill_between(_x[:2], _y[:2], _y[2:], color=fill)

        (draw,) = ax.plot(x_cycle, y_cycle, "-", color=draw, lw=2)
        # symbol = None
        if symbol is not None:
            symbol = ax.plot(pos, 0, color=symbol_color, marker=symbol, markersize=10)

        return draw, fill, symbol

    def init_grid(self):
        """For Render.

        Draw grid.

        :meta public:
        """
        self.draws = []
        self.fills = []
        self.symbols = []
        for i in range(self.gridsize):
            draw, fill, symbol = self.set_box(self.ax, i, shortcut="void")
            self.draws.append(draw)
            self.fills.append(fill)
            self.symbols.append(symbol)
        for t_array in self.state["targets"]:
            t = int(t_array)
            self.fills[t].remove()
            self.draws[t].remove()
            if self.symbols[t]:
                self.symbols[t].remove()
            draw, fill, symbol = self.set_box(self.ax, t, shortcut="target")
            self.draws[t] = draw
            self.fills[t] = fill
            self.symbols[t] = symbol
        t = int(self.bundle.game_state["user_state"]["goal"].squeeze().tolist())
        self.fills[t].remove()
        self.draws[t].remove()
        if self.symbols[t]:
            self.symbols[t][0].remove()
        draw, fill, symbol = self.set_box(self.ax, t, shortcut="goal")
        self.draws[t] = draw
        self.fills[t] = fill
        self.symbols[t] = symbol

        self.draw_pos = self.state["position"].squeeze().tolist()
        self.update_position()

        self.ax.set_yticks([])
        self.ax.set_xticks(list(range(self.gridsize))[::5])
        self.ax.set_xlim([-0.6, self.gridsize - 0.4])
        plt.show(block=False)

    def update_position(self):
        """For Render.

        Update the plot, changing the cursor position.

        :meta public:
        """
        t = int(self.draw_pos)
        self.fills[t].remove()
        self.draws[t].remove()

        if self.symbols[t]:
            try:
                self.symbols[t].remove()
            except TypeError:
                self.symbols[t][0].remove()
        if t == self.bundle.game_state["user_state"]["goal"]:
            shortcut = "goal"
        elif t in self.state["targets"]:  # maybe squeeze().tolist()
            shortcut = "target"
        else:
            shortcut = "void"
        draw, fill, symbol = self.set_box(self.ax, t, shortcut=shortcut)
        self.draws[t] = draw
        self.fills[t] = fill
        self.symbols[t] = symbol

        t = int(self.state["position"])
        draw, fill, symbol = self.set_box(self.ax, t, shortcut="position")
        self.draws[t] = draw
        self.fills[t] = fill
        self.symbols[t] = symbol
        self.draw_pos = t
