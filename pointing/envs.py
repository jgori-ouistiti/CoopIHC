import numpy
import matplotlib.pyplot as plt
from collections import OrderedDict
import gym
import core
from core.interactiontask import InteractionTask, PipeTaskWrapper
from core.space import StateElement
from core.helpers import sort_two_lists
from loguru import logger
import functools


class DiscretePointingTaskPipeWrapper(PipeTaskWrapper):

    def update_task_state(self, state):
        for key in self.state:
            try:
                value = state.pop(key)
                if key == 'position':
                    self.state[key]['values'] = numpy.array(value)
                elif key == 'targets':
                    self.state[key]['values'] = [numpy.array(v) for v in value]
            except KeyError:
                raise KeyError('Key "{}" defined in task state was not found in the received data'.format(key))
        if state:
            print("warning: the received data has not been consumed. {} does not match any current task state".format(str(state)))


    def update_operator_state(self, state):
        for key in state:
            try:
                self.bundle.operator.state[key]
            except KeyError:
                raise KeyError('Key "{}" sent in received data was not found in operator state'.format(key))
            self.bundle.operator.state[key]['values'] = state[key]


class DiscretePointingTask(InteractionTask):
    def __init__(self, dim = 2, gridsize = (31,31), number_of_targets = 10, mode = 'gain', layout = 'uniform'):
        super().__init__()
        self.dim = dim
        self.gridsize = gridsize
        self.number_of_targets = number_of_targets
        self.mode = mode
        self.layout = 'uniform'

        self.handbook['render_mode'].extend(['plot', 'text'])
        _dim = {"name": 'dim', 'value': dim, "meaning": 'dimension of the gridworld. Only 1 and 2 supported for now'}
        _gridsize = {"name": 'gridsize', "value": gridsize, "meaning": 'Size of the gridworld'}
        _number_of_targets = { "name": 'number_of_targets', "value": number_of_targets, 'meaning': 'number of potential targets from which to choose a goal'}
        _mode = { "name": 'mode','value': mode, 'meaning': "whether the assistant is expected to work as gain or as positioner. In the first case (gain), the operator's action is multiplied by the assistant's action to determine by how much to shift the old position of the cursor. In the second case (position) the assistant's action is directly the new position of the cursor."}
        self.handbook['parameters'].extend([_dim, _gridsize, _number_of_targets, _mode])


        self.state['position'] = StateElement(
                    values = None,
                    spaces = [core.space.Box(
                            low = numpy.array([0 for i in range(dim)]),
                            high = numpy.array([gridsize[i] for i in range(dim)]),
                            dtype = numpy.int8 )],
                    possible_values = None,
                    clipping_mode = 'clip'  )

        self.state['targets'] = StateElement(
                    values = None,
                    spaces = [[core.space.Box(
                            low = numpy.array([0 for i in range(dim)]),
                            high = numpy.array([gridsize[i] for i in range(dim)]),
                            dtype = numpy.int8 )] for j in range(self.number_of_targets)],
                    possible_values = None,
                    clipping_mode = 'error')


        self.Coordinates = functools.partial(self.GCoordinates, gridsize)

    class GCoordinates:
        def __init__(self, gridsize = None, index = None, coords = None):
            self.__index = None
            self.__coords = None
            if gridsize is None:
                raise ValueError("gridsize has to be provided to Coordinates __init__")
            self.__gridsize = gridsize
            if index is None and coords is None:
                raise ValueError("either index or coords have to be provided")
            if index is not None:
                self.__index = int(index)
            if coords is not None:
                self.__coords = tuple(coords)

        def index_to_coords(self, index):
            nl,nc = self.__gridsize
            ci = index%nc
            li = int((index-ci)/nc)
            return (li,ci)

        def coords_to_index(self, coords):
            li,ci = coords
            nl,nc = self.__gridsize
            return int(li*nc + ci)

        @property
        def coords(self):
            if not self.__coords:
                self.__coords = self.index_to_coords(self.__index)
            return self.__coords

        @coords.setter
        def coords(self, value):
            self.__coords = tuple(value)
            self.__index = self.coords_to_index(tuple(value))

        @property
        def index(self):
            if not self.__index:
                self.__index = self.coords_to_index(self.__coords)
            return self.__index

        @index.setter
        def index(self, value):
            self.__index = int(value)
            self.__coords = self.index_to_coords(int(value))


    def reset(self, dic = None):
        """ Reset the task.

        Reset the grid used for rendering, define new targets, select a starting position

        :param args: reset to the given state

        :meta public:
        """

        super().reset()
        self.grid = numpy.full(self.gridsize, ' ', dtype = numpy.str_)
        self.gridindex = numpy.arange(numpy.prod(self.gridsize)).reshape(self.gridsize)
        if self.layout == 'uniform':
            selected = numpy.random.choice(self.gridindex.reshape(-1), size = self.number_of_targets+1, replace = False)
            position = self.Coordinates(index = selected[0])
            targets = [self.Coordinates(index = v) for v in sorted(selected[1:])]
        else:
            raise NotImplementedError
        nl, nc = self.gridsize
        for i in targets:
            self.grid[i.coords] = 'T'

        self.state['position']['values'] = numpy.array(position.coords)
        self.state['targets']['values'] = [numpy.array(t.coords) for t in targets]

        if dic is not None:
            super().reset(dic = dic)


    def is_done_operator(self):
        # env ends when operator inputs action 0 or when too many turns have been played
        if self.turn > 50:
            return True
        if self.operator_action['values'][0] == 0:
            return True
        return False

    def operator_step(self, *args, **kwargs):
        state, sumrewards, is_done, reward_list = super().operator_step()
        return state, sumrewards, self.is_done_operator(), reward_list

    def is_done_assistant(self):
        return False

    def assistant_step(self, *args, **kwargs):
        super().assistant_step()

        if self.mode == 'position':
            self.state['position']['values'] = self.bundle.game_state['assistant_action']['action']['values']
        elif self.mode == 'gain':
            assistant_action = self.bundle.game_state['assistant_action']['action']['values'][0]
            operator_action = self.bundle.game_state['operator_action']['action']['human_values'][0]
            position = self.state['position']['values'][0]

            # Clipping and numpy dtype means we don't have to deal with edge cases, rounding etc.
            self.state['position']['values'] = position + operator_action*assistant_action

        return self.state, -1/2, self.is_done_assistant(), {}



    def render(self,*args, mode="text"):
        """ Render the task.

        Plot or print the grid, with cursor and target positions.

        :param ax:
        :param args:
        :param mode: 'text' or 'plot'


        :meta public:
        """




        if 'text' in mode:
            print('\n')
            print("Turn number {:f}".format(self.turn))
            print('Current position')
            print(self.state['position']['values'])
            print('Targets:')
            print(self.state['targets']['values'])
        else:
            raise NotImplementedError




class SimplePointingTask(InteractionTask):
    """ A 1D pointing task.

    A 1D grid of size 'Gridsize'. The cursor is at a certain 'position' and there are several potential 'targets' on the grid. The operator action is modulated by the assistant.

    :param gridsize: (int) Size of the grid
    :param number_of_targets: (int) Number of targets on the grid

    :meta public:
    """
    def __init__(self, gridsize = 31, number_of_targets = 10, mode = 'gain'):
        super().__init__()
        self.gridsize = gridsize
        self.number_of_targets = number_of_targets
        self.mode = mode
        self.dim = 1



        self.state['position'] = StateElement(
                    values = None,
                    spaces = [core.space.Discrete(gridsize)],
                    possible_values = None,
                    clipping_mode = 'clip'  )

        self.state['targets'] = StateElement(
                    values = None,
                    spaces = [core.space.Discrete(gridsize) for i in range(self.number_of_targets)], possible_values = None  )


    def reset(self, dic = None):
        """ Reset the task.

        Reset the grid used for rendering, define new targets, select a starting position

        :param args: reset to the given state

        :meta public:
        """

        super().reset()
        self.grid = [' ' for i in range(self.gridsize)]
        targets = sorted(numpy.random.choice(list(range(self.gridsize)), size = self.number_of_targets, replace = False))
        for i in targets:
            self.grid[i] = 'T'
        # Define starting position
        copy = list(range(len(self.grid)))
        for i in targets:
            copy.remove(i)
        position = int(numpy.random.choice(copy))
        self.state['position']['values'] = position
        self.state['targets']['values'] = targets
        if dic is not None:
            super().reset(dic = dic)


    def operator_step(self, *args, **kwargs):
        """ Do nothing, increment turns, return half a timestep

        :meta public:
        """
        return super().operator_step()

    def is_done_assistant(self):
        is_done = False
        if self.state['position']['values'][0] == self.bundle.game_state['operator_state']['goal']['values'][0]:
            is_done = True
        logger.info("Task {} done: {}".format(self.__class__.__name__, is_done))
        logger.info("Condition: Cursor position ({}) = Operator Goal ({})".format(str(self.state['position']['values'][0]), self.bundle.game_state['operator_state']['goal']['values'][0] ))
        return is_done

    def assistant_step(self, *args, **kwargs):
        """ Modulate the operator's action.

        Multiply the operator action with the assistant action.
        Update the position and grids.

        :param assistant_action: (list)

        :return: new state (OrderedDict), half a time step, is_done (True/False)

        :meta public:
        """
        super().assistant_step()
        is_done = False

        # Stopping condition if too many turns
        if self.turn > 50:
            return self.state, -1/2, True, {}

        if self.mode == 'position':
            self.state['position']['values'] = self.bundle.game_state['assistant_action']['action']['values']
        elif self.mode == 'gain':
            assistant_action = self.bundle.game_state['assistant_action']['action']['human_values'][0]
            operator_action = self.bundle.game_state['operator_action']['action']['human_values'][0]
            position = self.state['position']['human_values'][0]

            # self.state['position']['values'] = [int(numpy.clip(numpy.round(position + operator_action*assistant_action, decimals = 0), 0, self.gridsize-1))]
            self.state['position']['values'] = int(numpy.round(position + operator_action*assistant_action))


        return self.state, -1/2, self.is_done_assistant(), {}

    def render(self,*args, mode="text"):
        """ Render the task.

        Plot or print the grid, with cursor and target positions.

        :param ax:
        :param args:
        :param mode: 'text' or 'plot'

        .. warning::

            revisit the signature of this function

        :meta public:
        """
        goal = self.bundle.game_state['operator_state']['goal']['values'][0]

        self.grid[goal] = 'G'
        if 'text' in mode:
            tmp = self.grid.copy()
            tmp[int(self.state['position']['values'][0])] = 'P'
            _str = "|"
            for t in tmp:
                _str += t + "|"

            print('\n')
            print("Turn number {:f}".format(self.turn))
            print(_str)

            targets = sorted(self.state['targets']['values'])
            print('Targets:')
            print(targets)
            print("\n")
        if 'plot' in mode:
            axtask, axoperator, axassistant = args[:3]
            if self.ax is not None:
                self.update_position()
            else:
                self.ax = axtask
                self.init_grid()
                self.ax.set_aspect('equal')
        if not ('plot' in mode or 'text' in mode):
            raise NotImplementedError


    def set_box(self, ax, pos, draw = "k", fill = None, symbol = None, symbol_color = None, shortcut = None):
        """ For Render.

        Draws box on the given axis ax at the given position pos, with a given style.

        :meta public:
        """
        if shortcut == 'void':
            draw = 'k'
            fill = '#aaaaaa'
            symbol = None
        elif shortcut == 'target':
            draw = '#96006c'
            fill = "#913979"
            symbol = "$T$"
            symbol_color = 'k'
        elif shortcut == 'goal':
            draw = '#009c08'
            fill = '#349439'
            symbol = "$G$"
            symbol_color = 'k'
        elif shortcut == 'position':
            draw = '#00189c'
            fill = "#6573bf"
            symbol = "$P$"
            symbol_color = 'k'

        BOX_SIZE = 1
        BOX_HW = BOX_SIZE / 2

        _x = [pos-BOX_HW, pos+BOX_HW, pos + BOX_HW, pos - BOX_HW]
        _y = [-BOX_HW, -BOX_HW, BOX_HW, BOX_HW]
        x_cycle = _x + [_x[0]]
        y_cycle = _y + [_y[0]]
        if fill is not None:
            fill = ax.fill_between(_x[:2], _y[:2], _y[2:], color = fill)

        draw, = ax.plot(x_cycle,y_cycle, '-', color = draw, lw = 2)
        # symbol = None
        if symbol is not None:
            symbol = ax.plot(pos, 0, color = symbol_color, marker = symbol, markersize = 10)

        return draw, fill, symbol

    def init_grid(self):
        """ For Render.

        Draw grid.

        :meta public:
        """
        self.draws = []
        self.fills = []
        self.symbols = []
        for i in range(self.gridsize):
            draw, fill, symbol = self.set_box(self.ax, i, shortcut = 'void')
            self.draws.append(draw)
            self.fills.append(fill)
            self.symbols.append(symbol)
        for t in self.state['targets']['values']:
            self.fills[t].remove()
            self.draws[t].remove()
            if self.symbols[t]:
                self.symbols[t].remove()
            draw, fill, symbol = self.set_box(self.ax, t, shortcut = 'target')
            self.draws[t] = draw
            self.fills[t] = fill
            self.symbols[t] = symbol
        t = self.bundle.game_state['operator_state']['goal']['values'][0]
        self.fills[t].remove()
        self.draws[t].remove()
        if self.symbols[t]:
            self.symbols[t][0].remove()
        draw, fill, symbol = self.set_box(self.ax, t, shortcut = 'goal')
        self.draws[t] = draw
        self.fills[t] = fill
        self.symbols[t] = symbol

        self.draw_pos = self.state['position']['values'][0]
        self.update_position()



        self.ax.set_yticks([])
        self.ax.set_xticks(list(range(self.gridsize))[::5])
        self.ax.set_xlim([-.6,self.gridsize - .4])
        plt.show(block = False)

    def update_position(self):
        """ For Render.

        Update the plot, changing the cursor position.

        :meta public:
        """
        t = self.draw_pos
        self.fills[t].remove()
        self.draws[t].remove()

        if self.symbols[t]:
            try:
                self.symbols[t].remove()
            except TypeError:
                self.symbols[t][0].remove()
        if t == self.bundle.game_state['operator_state']['goal']['values'][0]:
            shortcut = 'goal'
        elif t in self.state['targets']['values']:
            shortcut = 'target'
        else:
            shortcut = 'void'
        draw, fill, symbol = self.set_box(self.ax, t, shortcut = shortcut)
        self.draws[t] = draw
        self.fills[t] = fill
        self.symbols[t] = symbol


        t = self.state['position']['values'][0]
        draw, fill, symbol = self.set_box(self.ax, t, shortcut = 'position')
        self.draws[t] = draw
        self.fills[t] = fill
        self.symbols[t] = symbol
        self.draw_pos = t


# class Continuous1DPointing(ClassicControlTask):
#     """ A 1D pointing task in Continuous space, unbounded, centered target
#
#     """
#
#     def __init__(self, I, b, ta, te):
#         a1 = b/(ta*te*I)
#         a2 = 1/(ta*te) + (1/ta + 1/te)*b/I
#         a3 = b/I + 1/ta + 1/te
#         bu = 1/(ta*te*I)
#
#
#         A = numpy.array([   [0, 1, 0, 0],
#                             [0, 0, 1, 0],
#                             [0, 0, 0, 1],
#                             [0, -a1, -a2, -a3]    ])
#
#         B = numpy.array([[ 0, 0, 0, bu]]).reshape((-1,1))
#
#         C = numpy.array([   [1, 0, 0, 0],
#                             [0, 1, 0, 0],
#                             [0, 0, 1, 0]
#                                 ])
#         super().__init__(4, .01, A, B, C)

# class CS_DT_Pointing(DT_ClassicControlTask):
#     """ A 1D pointing task in Continuous space, Discrete Time, with unbounded, centered target
#
#     """
#
#     def __init__(self, F, G, timestep = 0.1):
#
#
#
#         super().__init__(A.shape[1], timestep, A, B)



class Screen_v0(InteractionTask):
    def __init__(self, screen_size, number_of_targets, target_radius = 1e-2, max_cycle = 200, timestep = 0.1, mode = 'gain'):
        """ A 2D pointing.

        :param screen_size: (list) [width, height] in pixels
        :param number_of_targets: (int) The number of targets that will be placed on the screen
        """
        # Call super().__init__() before anything else
        super().__init__()
        # Task properties
        self.screen_size = screen_size
        self.aspect_ratio = self.screen_size[0]/self.screen_size[1]
        self.screen_low = numpy.array([-1, -1/self.aspect_ratio])
        self.screen_high = numpy.array([1, 1/self.aspect_ratio])
        self.number_of_targets = int(number_of_targets)
        self.max_cycle = max_cycle
        self.timestep = timestep
        self.target_radius = target_radius
        self.mode = mode


        self.handbook['render_mode'].extend(['plot', 'text'])
        _screen_size = {"name": 'screen_size', "value": screen_size, "meaning": 'Size of the screen. The aspect ratio will be computed from the screen size to calculate the height of the screen, the width being [-1,1]'}
        _number_of_targets = { "name": 'number_of_targets', "value": number_of_targets, 'meaning': 'number of potential targets from which to choose a goal'}
        _target_radius = { "name": 'target_radius', "value": target_radius, 'meaning': 'radius of the circular targets displayed on screen'}
        _max_cycle = { "name": 'max_cycle', "value": max_cycle, 'meaning': 'Maximum number of steps possible in this environment before it finished'}
        _timestep = { "name": 'timestep', "value": timestep, 'meaning': 'Duration of one step in this environment'}
        _mode = { "name": 'mode','value': mode, 'meaning': "whether the assistant is expected to work as gain or as positioner. In the first case (gain), the operator's action is multiplied by the assistant's action to determine by how much to shift the old position of the cursor. In the second case (position) the assistant's action is directly the new position of the cursor."}
        self.handbook['parameters'].extend([_screen_size, _number_of_targets, _target_radius, _max_cycle, _timestep, _mode])

        # Define state
        self.state['position'] = StateElement(
            values = numpy.array([0.1,0.1]),
            spaces = [  core.space.Box( low = self.screen_low, high = self.screen_high )  ],
            possible_values = None
                )

        self.state['targets'] = StateElement(
            values = [numpy.array([-1 + 2*i/self.number_of_targets, -1/self.aspect_ratio + i/self.aspect_ratio/self.number_of_targets]) for i in range(self.number_of_targets)],
            spaces = [  core.space.Box( low = self.screen_low, high = self.screen_high )  for i in range(self.number_of_targets)],
            possible_values = None
        )

        # Rendering stuff
        self.ax = None




    def reset(self, dic = None):
        """ Randomly draw N+1 disjoint disks. The N first disks are used as targets, the last disk's center is used as cursor position.
        """

        super().reset()

        targets = []
        radius = self.target_radius
        while len(targets)  < self.number_of_targets + 1: # generate targets as well as a start position
            candidate_target = self.state['position']['spaces'][0].sample()
            x,y = candidate_target
            passed = True
            if (x > -1 + 2*radius) and (x < 1 -radius) and (y > -1/self.aspect_ratio + 2*radius) and (y < 1/self.aspect_ratio - 2*radius):
                if not targets:
                    targets.append(candidate_target)
                else:
                    for xt,yt in targets:
                        if (xt-x)**2 + (yt-y)**2 <= 4*radius**2:
                            passed = False
                    if passed:
                        targets.append(candidate_target)

        position = targets.pop(-1)
        self.targets = targets
        radius = [numpy.sqrt(numpy.sum((target - numpy.array([-1,-1]))**2)) for target in targets]
        sorted_targets, sorted_radii = sort_two_lists(targets, radius, lambda x: x[1])

        self.state['position']['values'] = position
        self.state['targets']['values'] = sorted_targets

        if dic is not None:
            super().reset(dic = dic)



    def operator_step(self, *args, **kwargs):
        return super().operator_step()

    def assistant_step(self, assistant_action):
        super().assistant_step()
        is_done = False

        if self.turn > self.max_cycle:
            return self.state, -1/2, True, {}


        if self.mode == 'position':
            self.state['position']['values'] =  self.bundle.game_state['assistant_action']['action']['values']
        elif self.mode == 'gain':
            assistant_action = self.bundle.game_state['assistant_action']['action']['values'][0]
            operator_action = self.bundle.game_state['operator_action']['action']['values'][0]
            position = self.state['position']['values'][0]
            newpos = position + operator_action*assistant_action

            self.state['position']['values'] = numpy.clip(newpos, self.screen_low, self.screen_high)


        # if dist(self.bundle.operator.state['goal']['values'][0]) < self.target_radius:
        #     is_done = True

        return self.state, -1/2, self.is_done_assistant(), {}

    def is_done_assistant(self):
        is_done = False
        speed = 0
        threshold = 1
        goal = self.bundle.operator.state['goal']['values'][0]
        position = self.state['position']['values'][0]
        dist = numpy.sqrt(numpy.sum((position-goal)**2))
        if dist <= self.target_radius and speed < threshold:
            is_done = True

        logger.info("Task {} done: {}".format(self.__class__.__name__, is_done))
        logger.info("Condition: dist(Cursor position ({}) - Operator Goal ({}) ) = {} < target radius {}".format(str(self.state['position']['values'][0]), self.bundle.game_state['operator_state']['goal']['values'][0] ,str(dist), str(self.target_radius)))
        return is_done


    def render(self, *args, mode = 'plot'):
        axgame, axoperator, axassistant = args
        if self.ax is not None:
            self.update_position()
        else:
            self.ax = axgame
            self.init_grid(axoperator)
            self.ax.set_aspect('equal')
            plt.show(block = False)


    def init_grid(self, axoperator):
        targets = self.state['targets']['values']
        self.ax.set_xlim([-1.1 , 1.1])
        self.ax.set_ylim([-1.1/self.aspect_ratio , 1.1/self.aspect_ratio])

        # draw screen edge
        self.ax.plot([-1, 1, 1, -1, -1], [-1/self.aspect_ratio,-1/self.aspect_ratio, 1/self.aspect_ratio, 1/self.aspect_ratio, -1/self.aspect_ratio], lw = 1, color = 'k')

        for x,y in targets:
            circle = plt.Circle((x,y), self.target_radius, color = '#913979')
            self.ax.add_patch(circle)

        x,y = self.state['position']['values'][0]
        self.cursor =  plt.Circle((x,y), self.target_radius/5, color = 'r')
        self.ax.add_patch(self.cursor)
        cursor2 = plt.Circle((x,y), self.target_radius/5, color = 'r')
        axoperator.add_patch(cursor2)

    def update_position(self):
        self.cursor.remove()
        x,y = self.state['position']['values'][0]
        self.cursor =  plt.Circle((x,y), self.target_radius/5, color = 'r')
        self.ax.add_patch(self.cursor)


    #
    #
    #
    # def render(self,*args, mode="text"):
    #
    #     if 'text' in mode:
    #         pass
    #     if 'plot' in mode:
    #         axtask, axoperator, axassistant = args[:3]
    #         if self.ax is not None:
    #             self.update_position()
    #         else:
    #             self.ax = axtask
    #             self.init_grid(axoperator)
    #             self.ax.set_aspect('equal')
    #     if not ('plot' in mode or 'text' in mode):
    #         raise NotImplementedError
