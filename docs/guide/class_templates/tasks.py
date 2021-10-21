class SimplePointingTask(InteractionTask):
    """ A 1D pointing task.

    A 1D grid of size 'Gridsize'. The cursor is at a certain 'Position' and there are several potential 'Targets' on the grid. The user action is modulated by the assistant.

    :param gridsize: (int) Size of the grid
    :param number_of_targets: (int) Number of targets on the grid

    :meta public:
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.state['mystate'] = StateElement(
                    values = None,
                    spaces = [gym.spaces.Discrete(2)],
                    possible_values = None,
                    mode = 'error'  )



    def reset(self, dic = None):
        """ Reset the task.

        Reset the grid used for rendering, define new targets, select a starting position

        :param args: reset to the given state

        :meta public:
        """

        super().reset()

        # set state to some value
        self.state['mystate'] =

        if dic is not None:
            super().reset(dic = dic)

        return self.state

    def user_step(self, *args, **kwargs):

        super().user_step()

        is_done = False
        user_action = self.bundle.game_state['user_action']['action']
        assistant_action = self.bundle.game_state['assistant_action']['action']

        # set state to some value
        self.state['mystate'] =

        return self.state, reward, is_done, {}


    def assistant_step(self, *args, **kwargs):
        """ Modulate the user's action.

        Multiply the user action with the assistant action.
        Update the position and grids.

        :param assistant_action: (list)

        :return: new state (OrderedDict), half a time step, is_done (True/False)

        :meta public:
        """
        super().assistant_step()
        # do something
        is_done = False
        return self.state, -1/2, is_done, {}

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

        if 'text' in mode:
            # print something
        if 'plot' in mode:
            axtask, axuser, axassistant = args[:3]
            if self.ax is not None:
                # plot something
            else:
                self.ax = axtask
                self.ax.set_aspect('equal')
