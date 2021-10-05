from core.interactiontask import InteractionTask
from core.space import StateElement, Space, State

import numpy


# ================= Start defining task =======================

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
            print("Turn number {:f}".format(self.turn))
            print(self.state)
            print("\n")
        else:
            raise NotImplementedError

task = ExampleTask()

# ================= End defining task =======================



# ================= Test Task with _DevelopTask bundle
from core.bundle import _DevelopTask
from core.policy import BasePolicy
# Define agent policies
user_action_state = State()
user_action_state['action'] = StateElement( values = None, spaces = Space([numpy.array([-1,0,1], dtype = numpy.int16)]))

assistant_action_state = State()
assistant_action_state['action'] = StateElement( values = None, spaces = Space([numpy.array([-1,0,1], dtype = numpy.int16)]))

# Call _DevelopTask with agent policies provided (so that we can sample from it)
bundle = _DevelopTask(task,
    user_policy = BasePolicy(user_action_state),
    assistant_policy = BasePolicy(assistant_action_state)
    )

# Reset the task, plot the state.
bundle.reset()
bundle.render("text")
# Test simple input
bundle.step([numpy.array([1]),numpy.array([1])])
bundle.render("text")

# Test with input sampled from the agent policies
bundle.reset()
while True:
    ret_user, ret_assistant = bundle.step([bundle.user.policy.sample()[0], bundle.assistant.policy.sample()[0]])
    print(ret_user)
    is_done = ret_user[2] or ret_assistant[2]
    if is_done:
        break
