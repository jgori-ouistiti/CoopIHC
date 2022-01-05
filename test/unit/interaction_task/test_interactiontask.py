"""This module provides tests for the InteractionTask class of the
coopihc package."""


import numpy
from coopihc import InteractionTask, StateElement, Space, discrete_space
from coopihc.space.utils import StateNotContainedWarning, StateNotContainedError
import pytest


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
            0, discrete_space(numpy.array([-1, 0, 1])), out_of_bounds_mode="warning"
        )


class MinimalTaskWithStateAugmented(MinimalTask):
    """Non-functional minimal subclass including a more complex
    state to use in tests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state["x"] = StateElement(
            0, discrete_space(numpy.array([-1, 0, 1])), out_of_bounds_mode="warning"
        )
        self.state["y"] = StateElement(
            2, discrete_space(numpy.array([2, 3, 4, 5])), out_of_bounds_mode="error"
        )
        self.state["z"] = StateElement(
            2,
            discrete_space(numpy.array([i for i in range(1, 10)])),
            out_of_bounds_mode="clip",
        )


class MinimalTaskWithStateAndDirectReset(MinimalTaskWithState):
    """Non-functional minimal subclass including a state and reset method
    to use in tests. This class resets the state element directly."""

    def reset(self, dic=None):
        reset_value = -1
        self.state["x"][:] = reset_value


class MinimalTaskWithStateAndResetViaState(MinimalTaskWithState):
    """Non-functional minimal subclass including a state and reset method
    to use in tests. This class resets the state element via the state
    property."""

    def reset(self, dic=None):
        reset_value = -1
        self.state.reset(dic={"x": reset_value})


class MinimalTaskWithStateAndResetViaStateElement(MinimalTaskWithState):
    """Non-functional minimal subclass including a state and reset method
    to use in tests. This class resets the state element via the
    StateElement's reset method."""

    def reset(self, dic=None):
        reset_value = -1
        self.state["x"].reset(value=reset_value)


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
        value = task.state["x"].squeeze().tolist()
        counter[value] += 1

    for value in possible_values:
        assert counter[value] > 0


def test_base_reset_without_dic():
    """Tests the reset method when no dic is provided."""
    test_double_base_reset_without_dic()
    test_base_reset_randomness()


def test_base_reset_with_full_dic():
    task = MinimalTaskWithState()
    reset_dic = {"x": numpy.array([0])}
    task._base_reset(reset_dic)
    assert isinstance(task.state["x"], StateElement)
    assert task.state["x"] == 0
    reset_dic = {"x": numpy.array([1])}
    task._base_reset(reset_dic)
    assert isinstance(task.state["x"], StateElement)
    assert task.state["x"] == 1
    reset_dic = {"x": numpy.array([-1])}
    task._base_reset(reset_dic)
    assert isinstance(task.state["x"], StateElement)
    assert task.state["x"] == -1
    reset_dic = {"x": numpy.array([-2])}
    with pytest.warns(StateNotContainedWarning):
        task._base_reset(reset_dic)
    assert isinstance(task.state["x"], StateElement)
    assert task.state["x"] == -2
    reset_dic = {"x": numpy.array([2])}
    with pytest.warns(StateNotContainedWarning):
        task._base_reset(reset_dic)
    assert task.state["x"] == 2
    assert isinstance(task.state["x"], StateElement)
    task = MinimalTaskWithStateAugmented()
    reset_dic = {"x": numpy.array([0]), "y": numpy.array([5]), "z": numpy.array([1])}
    task._base_reset(reset_dic)
    assert task.state["x"] == 0
    assert isinstance(task.state["x"], StateElement)
    assert task.state["y"] == 5
    assert isinstance(task.state["y"], StateElement)
    assert task.state["z"] == 1
    assert isinstance(task.state["z"], StateElement)
    reset_dic = {"x": numpy.array([0]), "y": numpy.array([6]), "z": numpy.array([1])}
    with pytest.raises(StateNotContainedError):
        task._base_reset(reset_dic)
    reset_dic = {"x": numpy.array([0]), "y": numpy.array([5]), "z": numpy.array([-8])}
    task._base_reset(reset_dic)
    assert task.state["z"] == 1


def test_base_reset_with_partial_dic():
    task = MinimalTaskWithStateAugmented()
    reset_dic = {"x": numpy.array([0]), "y": numpy.array([2])}
    task._base_reset(reset_dic)
    assert task.state["x"] == 0
    assert isinstance(task.state["x"], StateElement)
    assert task.state["y"] == 2
    assert isinstance(task.state["y"], StateElement)

    set_z = {}
    for i in range(100):
        task._base_reset(reset_dic)
        set_z[str(task.state["z"])] = task.state["z"].squeeze().tolist()

    assert sorted(list(set_z.values())) == [i for i in range(1, 10)]


def test_base_reset_with_overwritten_reset():
    """Tests the _base_reset method if the subclassed InteractionTask has
    implemented a custom reset methd."""
    for task_class in [
        MinimalTaskWithStateAndDirectReset,
        MinimalTaskWithStateAndResetViaState,
        MinimalTaskWithStateAndResetViaStateElement,
    ]:
        task = task_class()
        assert task.state["x"] == 0
        assert isinstance(task.state["x"], StateElement)
        task._base_reset()
        assert task.state["x"] == -1
        assert isinstance(task.state["x"], StateElement)


def test_base_reset():
    """Tests the forced reset mechanism provided by the _base_reset method"""
    test_base_reset_without_dic()
    test_base_reset_with_full_dic()
    test_base_reset_with_partial_dic()
    test_base_reset_with_overwritten_reset()


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
