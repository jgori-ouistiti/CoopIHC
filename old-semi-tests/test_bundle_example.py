def test_bundle_example():
    """Runs the bundle examples."""

    from coopihc.interactiontask import ExampleTask
    from coopihc.base import State, StateElement, Space
    from coopihc.bundle import Bundle
    from coopihc.agents import BaseAgent, ExampleUser
    from coopihc.policy import BasePolicy
    import numpy

    # [start-check-task]
    # Define agent action states (what actions they can take)
    user_action_state = State()
    user_action_state["action"] = StateElement(
        values=None,
        spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])],
    )
    user_policy = BasePolicy(user_action_state)
    user = BaseAgent("user", agent_policy=user_policy)

    assistant_action_state = State()
    assistant_action_state["action"] = StateElement(
        values=None,
        spaces=[Space([numpy.array([-1, 0, 1], dtype=numpy.int16)])],
    )
    assistant_policy = BasePolicy(assistant_action_state)
    assistant = BaseAgent("assistant", agent_policy=assistant_policy)

    # Bundle a task together with two BaseAgents
    bundle = Bundle(task=ExampleTask(), user=user, assistant=assistant)

    # Reset the task, plot the state.
    bundle.reset(turn=1)
    # print(bundle.game_state)
    # return user, assistant, user_policy, assistant_policy, bundle
    bundle.step(numpy.array([1]), numpy.array([1]))
    # print(bundle.game_state)

    # Test simple input
    bundle.step(numpy.array([1]), numpy.array([1]))

    # Test with input sampled from the agent policies
    bundle.reset()
    while True:
        task_state, rewards, is_done = bundle.step(
            bundle.user.policy.sample()[0], bundle.assistant.policy.sample()[0]
        )
        # print(task_state)
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
        # print(state, rewards, is_done)
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
        # print(state, rewards, is_done)
        if is_done:
            break
    # [end-highlevel-code]

    print("passed all")


if __name__ == "__main__":
    test_bundle_example()
