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
            print("Turn number {:f}".format(self.turn_number.squeeze().tolist()))
            print(self.state)
            print("\n")
        else:
            raise NotImplementedError


# ================= End defining task =======================



# ================= Test Task =======================
from core.bundle import Bundle
from core.policy import BasePolicy
from core.agents import BaseAgent

# Define agent action states (what actions they can take)
user_action_state = State()
user_action_state['action'] = StateElement(
    values = None,
    spaces = [Space([numpy.array([-1,0,1], dtype = numpy.int16)])]
    )

assistant_action_state = State()
assistant_action_state['action'] = StateElement(
    values = None,
    spaces = [Space([numpy.array([-1,0,1], dtype = numpy.int16)])]
    )

# Run a task together with two BaseAgents
bundle = Bundle(
    task = ExampleTask(),
    user = BaseAgent( 'user',
        override_agent_policy = BasePolicy(user_action_state)),
    assistant = BaseAgent( 'assistant',
        override_agent_policy = BasePolicy(assistant_action_state))
    )

# Reset the task, plot the state.
bundle.reset(turn = 1)
print(bundle.game_state)
bundle.step(numpy.array([1]),numpy.array([1]))
print(bundle.game_state)


# Test simple input
bundle.step(numpy.array([1]),numpy.array([1]))

# Test with input sampled from the agent policies
bundle.reset()
while True:
    task_state, rewards, is_done = bundle.step(bundle.user.policy.sample()[0], bundle.assistant.policy.sample()[0])
    print(task_state)
    if is_done:
        break
