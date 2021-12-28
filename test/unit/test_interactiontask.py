"""This module provides tests for the InteractionTask class of the
coopihc package."""


import numpy
from coopihc import InteractionTask, StateElement, Space


class MinimalTask(InteractionTask):
    """Non-functional minimal subclass to use in tests."""

    def user_step(self):
        pass

    def assistant_step(self):
        pass

    def reset(self, dic=None):
        pass


class MinimalTaskWithState(MinimalTask):
    """Non-functional minimal subclass including a state to use
    in tests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state["x"] = StateElement(
            values=0, spaces=Space([numpy.array([-1, 0, 1])])
        )


def test_imports():
    """Tests the different import ways for the InteractionTask."""
    from coopihc import InteractionTask
    from coopihc.interactiontask import InteractionTask
    from coopihc.interactiontask.InteractionTask import InteractionTask


def test_example():
    """Tries to import and create the example task."""
    from coopihc import ExampleTask

    ExampleTask()


def test_init():
    """Tries to initialize an InteractionTask and checks the expected
    properties and methods."""
    assert empty_init_fails()
    assert cant_be_subclassed_without_necessary_overrides()
    assert can_be_subclassed_with_minimal_overrides()
    test_properties()
    test_methods()


def test_properties():
    """Tests the expected properties for a minimal InteractionTask."""
    task = MinimalTask()
    # Direct attributes
    assert hasattr(task, "_state")
    assert hasattr(task, "bundle")
    assert hasattr(task, "round")
    assert hasattr(task, "timestep")
    assert hasattr(task, "ax")
    # Property functions
    # assert hasattr(task, "turn_number") # Only available with bundle
    assert hasattr(task, "state")
    assert hasattr(task, "user_action")
    assert hasattr(task, "assistant_action")


def test_methods():
    """Tests the expected methods for a minimal InteractionTask."""
    task = MinimalTask()
    # Public methods
    assert hasattr(task, "finit")
    assert hasattr(task, "base_user_step")
    assert hasattr(task, "base_assistant_step")
    assert hasattr(task, "user_step")
    assert hasattr(task, "assistant_step")
    assert hasattr(task, "reset")
    assert hasattr(task, "render")
    # Private methods
    assert hasattr(task, "__content__")
    assert hasattr(task, "_base_reset")


def can_be_subclassed_with_minimal_overrides():
    """Returns True if trying to subclass an InteractionTask with
    only overrides for user_step, assistant_step and reset succeeds."""
    MinimalTask()
    return True


def cant_be_subclassed_without_necessary_overrides():
    """Returns True if trying to subclass an InteractionTask without
    the necessary method overrides fails."""
    assert cant_be_subclassed_without_user_step()
    assert cant_be_subclassed_without_assistent_step()
    assert cant_be_subclassed_without_reset()
    return True


def cant_be_subclassed_without_reset():
    """Returns True if trying to subclass an InteractionTask without
    a reset method override fails."""

    class TaskWithoutReset(InteractionTask):
        def assistant_step(self):
            pass

        def user_step(self):
            pass

    try:
        TaskWithoutReset()
    except TypeError:
        return True


def cant_be_subclassed_without_user_step():
    """Returns True if trying to subclass an InteractionTask without
    a user_step method override fails."""

    class TaskWithoutAsssistentStep(InteractionTask):
        def assistant_step(self):
            pass

        def reset(self):
            pass

    try:
        TaskWithoutAsssistentStep()
    except TypeError:
        return True


def cant_be_subclassed_without_assistent_step():
    """Returns True if trying to subclass an InteractionTask without
    an assistent_step method override fails."""

    class TaskWithoutUserStep(InteractionTask):
        def user_step(self):
            pass

        def reset(self):
            pass

    try:
        TaskWithoutUserStep()
    except TypeError:
        return True


def empty_init_fails():
    """Returns True if trying to initialize an InteractionTask
    without any arguments fails."""
    try:
        InteractionTask()
    except TypeError:
        return True


def test_double_base_reset_without_dic():
    """Creates a minimal task and calls base reset on it twice."""
    task = MinimalTaskWithState()
    task._base_reset()
    task._base_reset()


def test_base_reset_randomness():
    """Tests that state value is set to random value within space when
    no dic is supplied."""
    task = MinimalTaskWithState()
    # Reset task state (should be random)
    possible_values = [-1, 0, 1]
    counter = {value: 0 for value in possible_values}

    for _ in range(1000):
        task._base_reset()
        value = task.state["x"].values[0][0][0]
        counter[value] += 1

    for value in possible_values:
        assert counter[value] > 0


def test_base_reset_without_dic():
    """Tests the reset method when no dic is provided."""
    test_double_base_reset_without_dic()
    test_base_reset_randomness()


def test_base_reset():
    """Tests the forced reset mechanism provided by the _base_reset method
    of InteractionTask that is called by _Bundle.reset()."""
    # Check reset without dic
    test_base_reset_without_dic()

    """The first thing that comes to mind that is missing is to verify the forced-reset mechanism. When you reset a bundle, you can pass it a reset_dictionnary {'task_state': task_reset_dic, 'user_state': user_reset_dic, etc.}. The value associated with the task_state key is passed to _base_reset as dic. A couple things to check here are that without dic provided, the state is randomly reset (maybe check by fixing the seed of the stateElements), with dic provided the reset is correctly achieved to the forced state in input, and in between (if the reset_dic specifies a value for x_1 but not x_2, x_2 should be reset randomly)"""


def test_interactiontask():
    """Tests the methods provided by the InteractionTask class."""
    test_imports()
    test_example()
    test_init()
    test_base_reset()


# +----------------------+
# +        MAIN          +
# +----------------------+
if __name__ == "__main__":
    test_interactiontask()
