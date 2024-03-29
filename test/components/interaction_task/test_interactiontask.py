"""This module provides tests for the InteractionTask class of the
coopihc package."""


import numpy
from coopihc import InteractionTask, StateElement
from coopihc.base.elements import array_element, cat_element, discrete_array_element
from coopihc.base.utils import StateNotContainedWarning, StateNotContainedError
import pytest


class MinimalTask(InteractionTask):
    """Non-functional minimal subclass to use in tests."""

    def on_user_action(self):
        pass

    def on_assistant_action(self):
        pass

    def reset(self, dic=None):
        pass


class MinimalTaskWithState(MinimalTask):
    """Non-functional minimal subclass including a state to use
    in tests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state["x"] = discrete_array_element(low=-1, high=1, init=0)


class MinimalTaskWithStateAugmented(MinimalTask):
    """Non-functional minimal subclass including a more complex
    state to use in tests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state["x"] = discrete_array_element(low=-1, high=1, init=0)
        self.state["y"] = discrete_array_element(
            low=2, high=5, init=2, out_of_bounds_mode="error"
        )
        self.state["z"] = discrete_array_element(
            low=0, high=9, init=0, out_of_bounds_mode="clip"
        )


class MinimalTaskWithStateAndDirectReset(MinimalTaskWithState):
    """Non-functional minimal subclass including a state and reset method
    to use in tests. This class resets the state element directly."""

    def reset(self, dic=None):
        reset_value = -1
        self.state["x"][...] = reset_value


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
    assert hasattr(task, "timestep")
    assert hasattr(task, "ax")
    # Property functions
    assert hasattr(task, "state")


def test_methods():
    """Tests the expected methods for a minimal InteractionTask."""
    task = MinimalTask()
    # Public methods
    assert hasattr(task, "finit")
    assert hasattr(task, "base_on_user_action")
    assert hasattr(task, "base_on_assistant_action")
    assert hasattr(task, "on_user_action")
    assert hasattr(task, "on_assistant_action")
    assert hasattr(task, "reset")
    assert hasattr(task, "render")
    # Private methods
    assert hasattr(task, "__content__")
    assert hasattr(task, "_base_reset")


def can_be_subclassed_with_minimal_overrides():
    """Returns True if trying to subclass an InteractionTask with
    only overrides for on_user_action, on_assistant_action and reset succeeds."""
    MinimalTask()
    return True


def cant_be_subclassed_without_necessary_overrides():
    """Returns True if trying to subclass an InteractionTask without
    the necessary method overrides fails."""
    assert cant_be_subclassed_without_on_user_action()
    assert cant_be_subclassed_without_assistent_step()
    assert cant_be_subclassed_without_reset()
    return True


def cant_be_subclassed_without_reset():
    """Returns True if trying to subclass an InteractionTask without
    a reset method override fails."""

    class TaskWithoutReset(InteractionTask):
        def on_assistant_action(self):
            pass

        def on_user_action(self):
            pass

    try:
        TaskWithoutReset()
    except TypeError:
        return True


def cant_be_subclassed_without_on_user_action():
    """Returns True if trying to subclass an InteractionTask without
    a on_user_action method override fails."""

    class TaskWithoutAsssistentStep(InteractionTask):
        def on_assistant_action(self):
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
        def on_user_action(self):
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
    task._base_reset(dic=reset_dic)
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
    task._base_reset(dic=reset_dic)
    assert task.state["x"] == 0
    assert isinstance(task.state["x"], StateElement)
    assert task.state["y"] == 5
    assert isinstance(task.state["y"], StateElement)
    assert task.state["z"] == 1
    assert isinstance(task.state["z"], StateElement)
    reset_dic = {"x": numpy.array([0]), "y": numpy.array([6]), "z": numpy.array([1])}
    with pytest.raises(StateNotContainedError):
        task._base_reset(dic=reset_dic)
    reset_dic = {"x": numpy.array([0]), "y": numpy.array([5]), "z": numpy.array([-8])}
    task._base_reset(dic=reset_dic)
    assert task.state["z"] == 0


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
        set_z[str(task.state["z"])] = task.state["z"].tolist()

    assert sorted(list(set_z.values())) == [i for i in range(10)]


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
