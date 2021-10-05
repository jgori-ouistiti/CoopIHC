from core.interactiontask import InteractionTask
from core.space import StateElement, Space, State

import numpy


class ExampleTask(InteractionTask):
    """ExampleTask with two agents. The task is finished when 'x' reaches 4. The user and assistants can both do -1, +0, +1 to 'x'.
    """
    def __init__(self, *args, **kwargs):
        # Call super().__init__() beofre anything else, which initializes some useful attributes, including a State (self.state) for the task
        super().__init__(*args, **kwargs)

        self.state['x'] = StateElement(
            values = 0,
            spaces = Space([
                numpy.array([-4,-3,-2,-1,0,1,2,3,4], dtype = numpy.int16)
            ]),
            clipping_mode = 'clip'
        )


    def reset(self, dic = None):
        self.state['x']['values'] = numpy.array([0])
        return


    def user_step(self, *args, **kwargs):
        is_done = False
        self.state['x'] += self.user_action
        if int(self.state['x']['values'][0]) == 4:
            is_done = True
        return self.state, -1, is_done, {}

    def assistant_step(self, *args, **kwargs):
        is_done = False
        self.state['x'] += self.assistant_action
        if int(self.state['x']['values'][0]) == 4:
            is_done = True
        return self.state, -1, is_done, {}

    def render(self,*args, mode="text"):
        if 'text' in mode:
            print('\n')
            print("Turn number {:f}".format(self.turn_number.squeeze().tolist()))
            print(self.state)
            print("\n")
        else:
            raise NotImplementedError
