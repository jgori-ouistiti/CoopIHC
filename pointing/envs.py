import numpy
import matplotlib.pyplot as plt
from collections import OrderedDict
import gym
from core.interactiontask import InteractionTask

class SimplePointingTask(InteractionTask):
    def __init__(self, gridsize = 31, number_of_targets = 10):
        super().__init__()
        self.state = OrderedDict({'Gridsize': [gridsize], 'Position': [2], 'Targets': [3,4]})
        self.number_of_targets = 10

    def reset(self, *args):
        if args:
            raise NotImplementedError
        else:
            self.grid = [' ' for i in range(self.state['Gridsize'][0])]
            targets = sorted(numpy.random.choice(list(range(self.state['Gridsize'][0])), size = self.number_of_targets, replace = False))
            for i in targets:
                self.grid[i] = 'T'
            # Define starting position
            copy = list(range(len(self.grid)))
            for i in targets:
                copy.remove(i)
            position = int(numpy.random.choice(copy))
            self.state['Position'] = [position]
            self.state['Targets'] = targets


    def operator_step(self, operator_action):
        self.turn += 1/2
        return self.state, -1/2, False, {}

    def assistant_step(self, assistant_action):
        self.turn += 1/2
        self.round += 1
        is_done = False
        operator_action = self.bundle.assistant.state['OperatorAction'][0]
        assistant_action = assistant_action[0]
        self.state['Position'] = [int(numpy.clip(numpy.round(self.state['Position'][0] + operator_action*assistant_action, decimals = 0), 0, self.state['Gridsize'][0]-1))]
        if self.state['Position'][0] == self.bundle.operator.state['Goal'][0]:
            is_done = True
        return self.state, -1/2, is_done, {}

    def render(self, ax, *args, mode="text"):
        goal = self.bundle.operator.state['Goal'][0]
        self.grid[goal] = 'G'
        if 'text' in mode:
            tmp = self.grid.copy()
            tmp[int(self.state['Position'][0])] = 'P'
            _str = "|"
            for t in tmp:
                _str += t + "|"

            print('\n')
            print("Turn number {:f}".format(self.turn))
            print(_str)

            targets = sorted(self.state['Targets'])
            print('Targets:')
            print(targets)
            print("\n")
        if 'plot' in mode:
            if self.ax is not None:
                self.update_position()
            else:
                self.ax = ax
                self.init_grid()
                self.ax.set_aspect('equal')
        if not ('plot' in mode or 'text' in mode):
            raise NotImplementedError


    def set_box(self, ax, pos, draw = "k", fill = None, symbol = None, symbol_color = None, shortcut = None):
        if shortcut == 'void':
            draw = 'k'
            fill = '#aaaaaa'
            symbol = None
        elif shortcut == 'target':
            draw = '#96006c'
            fill = "#913979"
            symbol = "1"
            symbol_color = 'k'
        elif shortcut == 'goal':
            draw = '#009c08'
            fill = '#349439'
            symbol = "X"
            symbol_color = 'k'
        elif shortcut == 'position':
            draw = '#00189c'
            fill = "#6573bf"
            symbol = "X"
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
        symbol = None
        if symbol is not None:
            symbol = ax.plot(pos, 0, color = symbol_color, marker = symbol, markersize = 100)

        return draw, fill, symbol

    def init_grid(self):
        self.draws = []
        self.fills = []
        self.symbols = []
        for i in range(self.state['Gridsize'][0]):
            draw, fill, symbol = self.set_box(self.ax, i, shortcut = 'void')
            self.draws.append(draw)
            self.fills.append(fill)
            self.symbols.append(symbol)
        for t in self.state['Targets']:
            self.fills[t].remove()
            self.draws[t].remove()
            if self.symbols[t]:
                self.symbols[t].remove()
            draw, fill, symbol = self.set_box(self.ax, t, shortcut = 'target')
            self.draws[t] = draw
            self.fills[t] = fill
            self.symbols[t] = symbol
        t = self.bundle.operator.state['Goal'][0]
        self.fills[t].remove()
        self.draws[t].remove()
        if self.symbols[t]:
            self.symbols[t].remove()
        draw, fill, symbol = self.set_box(self.ax, t, shortcut = 'goal')
        self.draws[t] = draw
        self.fills[t] = fill
        self.symbols[t] = symbol

        self.draw_pos = self.state["Position"][0]
        self.update_position()



        self.ax.set_yticks([])
        self.ax.set_xticks(list(range(self.state['Gridsize'][0]))[::5])
        self.ax.set_xlim([-.6,self.state['Gridsize'][0] - .4])
        plt.show(block = False)

    def update_position(self):
        t = self.draw_pos
        self.fills[t].remove()
        self.draws[t].remove()
        if self.symbols[t]:
            self.symbols[t].remove()
        if t == self.bundle.operator.state['Goal'][0]:
            shortcut = 'goal'
        elif t in self.state['Targets']:
            shortcut = 'target'
        else:
            shortcut = 'void'
        draw, fill, symbol = self.set_box(self.ax, t, shortcut = shortcut)
        self.draws[t] = draw
        self.fills[t] = fill
        self.symbols[t] = symbol


        t = self.state['Position'][0]
        draw, fill, symbol = self.set_box(self.ax, t, shortcut = 'position')
        self.draws[t] = draw
        self.fills[t] = fill
        self.symbols[t] = symbol
        self.draw_pos = t
