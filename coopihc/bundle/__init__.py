from ._Bundle import _Bundle
from .Bundle import Bundle
from .ModelChecks import ModelChecks
from .PlayAssistant import PlayAssistant
from .PlayBoth import PlayBoth
from .PlayNone import PlayNone
from .PlayUser import PlayUser
from .SinglePlayUser import SinglePlayUser
from .SinglePlayUserAuto import SinglePlayUserAuto
from .Train import Train

# Import wrappers for convenience
from .wrappers import BundleWrapper, PipedTaskBundleWrapper

from coopihc.space import State, StateElement, Space
from coopihc.agents import BaseAgent, ExampleUser
from coopihc.policy import BasePolicy

import numpy


# List of kwargs for bundles init()
#    - reset_skip_first_half_step (if True, skips the first_half_step of the bundle on reset. The idea being that the internal state of the agent provided during initialization should not be updated during reset). To generate a consistent observation, what we do is run the observation engine, but without potential noisefactors.


## Wrappers
# ====================


## =====================
## Train

# https://stackoverflow.com/questions/1012185/in-python-how-do-i-index-a-list-with-another-list/1012197
#
# class Flexlist(list):
#     def __getitem__(self, keys):
#         if isinstance(keys, (int, numpy.int, slice)): return list.__getitem__(self, keys)
#         return [self[k] for k in keys]
#
# class Flextuple(tuple):
#     def __getitem__(self, keys):
#         if isinstance(keys, (int, numpy.int, slice)): return tuple.__getitem__(self, keys)
#         return [self[k] for k in keys]


# ====================== Examples ==================
if __name__ == "__main__":
    from coopihc.interactiontask import ExampleTask

    # [start-check-task]
    # Define agent action states (what actions they can take)
    user_action_state = State()
    user_action_state["action"] = StateElement(
        values=None,
        spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])],
    )

    assistant_action_state = State()
    assistant_action_state["action"] = StateElement(
        values=None,
        spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])],
    )

    # Bundle a task together with two BaseAgents
    bundle = Bundle(
        task=ExampleTask(),
        user=BaseAgent("user", override_agent_policy=BasePolicy(user_action_state)),
        assistant=BaseAgent(
            "assistant",
            override_agent_policy=BasePolicy(assistant_action_state),
        ),
    )

    # Reset the task, plot the state.
    bundle.reset(turn=1)
    print(bundle.game_state)
    bundle.step(numpy.array([1]), numpy.array([1]))
    print(bundle.game_state)

    # Test simple input
    bundle.step(numpy.array([1]), numpy.array([1]))

    # Test with input sampled from the agent policies
    bundle.reset()
    while True:
        task_state, rewards, is_done = bundle.step(
            bundle.user.policy.sample()[0], bundle.assistant.policy.sample()[0]
        )
        print(task_state)
        if is_done:
            break
    # [end-check-task]

    # [start-check-taskuser]
    class ExampleTaskWithoutAssistant(ExampleTask):
        def assistant_step(self, *args, **kwargs):
            return self.state, 0, False, {}

    example_task = ExampleTaskWithoutAssistant()
    example_user = ExampleUser()
    bundle = Bundle(task=example_task, user=example_user)
    bundle.reset(turn=1)
    while 1:
        state, rewards, is_done = bundle.step(bundle.user.policy.sample()[0])
        print(state, rewards, is_done)
        if is_done:
            break
    # [end-check-taskuser]

    # [start-highlevel-code]
    # Define a task
    example_task = ExampleTask()
    # Define a user
    example_user = ExampleUser()
    # Define an assistant
    example_assistant = BaseAgent("assistant")
    # Bundle them together
    bundle = Bundle(task=example_task, user=example_user)
    # Reset the bundle (i.e. initialize it to a random or presecribed states)
    bundle.reset(turn=1)
    # Step through the bundle (i.e. play a full round)
    while 1:
        state, rewards, is_done = bundle.step(bundle.user.policy.sample()[0])
        print(state, rewards, is_done)
        if is_done:
            break
    # [end-highlevel-code]


#
#
# class _DevelopTask(Bundle):
#     """ A bundle without user or assistant. It can be used when developping tasks to facilitate some things like rendering
#     """
#     def __init__(self, task, **kwargs):
#
#         agents = []
#         for role in ['user', 'assistant']:
#             agent = kwargs.get(role)
#             if agent is None:
#                 agent_kwargs = {}
#                 agent_policy = kwargs.get("{}_policy".format(role))
#                 self.agent_policy = agent_policy
#                 if agent_policy is not None:
#                     agent_kwargs['policy'] = agent_policy
#                 agent = getattr(coopihc.agents, "Dummy"+role.capitalize())(**agent_kwargs)
#             else:
#                 kwargs.pop(agent)
#
#             agents.append(agent)
#         self.agents = agents
#
#
#         super().__init__(task, *agents, **kwargs)
#
#     def reset(self, dic = {}, **kwargs):
#         super().reset(dic = dic, **kwargs)
#
#     def step(self, joint_action):
#         user_action, assistant_action = joint_action
#         if isinstance(user_action, StateElement):
#             user_action = user_action['values']
#         if isinstance(assistant_action, StateElement):
#             assistant_action = assistant_action['values']
#         self.game_state["assistant_action"]['action']['values'] = assistant_action
#         self.game_state['user_action']['action']['values'] = user_action
#         ret_user = self.task.base_user_step(user_action)
#         ret_assistant = self.task.base_assistant_step(assistant_action)
#         return ret_user, ret_assistant
