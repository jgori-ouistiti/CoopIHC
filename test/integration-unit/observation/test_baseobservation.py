"""This module provides tests for the BaseObservationEngine class of the
coopihc package."""

from coopihc.observation.BaseObservationEngine import BaseObservationEngine
import pytest


class MinimalEngine(BaseObservationEngine):
    """Non-functional minimal subclass to use in tests."""


def test_imports():
    """Tests the different import ways for the Interactionobseng."""
    from coopihc import BaseObservationEngine
    from coopihc.observation import BaseObservationEngine
    from coopihc.observation.BaseObservationEngine import BaseObservationEngine


def test_example():
    """Tries to import and create the example obseng."""
    from coopihc import ExampleObservationEngine

    ExampleObservationEngine("substate")


def test_init():
    """Tries to initialize an Interactionobseng and checks the expected
    properties and methods."""
    assert empty_init()
    assert can_be_subclassed_without_overrides()
    test_properties()
    test_methods()


def test_properties():
    """Tests the expected properties for a minimal Interactionobseng."""
    obseng = MinimalEngine()
    # Property functions
    # assert hasattr(obseng, "turn_number") # Only available with bundle
    assert hasattr(obseng, "observation")
    assert hasattr(obseng, "action")


def test_methods():
    """Tests the expected methods for a minimal Interactionobseng."""
    obseng = MinimalEngine()
    # Public methods
    assert hasattr(obseng, "observe")
    assert hasattr(obseng, "reset")
    # Private methods
    assert hasattr(obseng, "__content__")


def can_be_subclassed_without_overrides():
    """Returns True if trying to subclass an Interactionobseng with
    only overrides for user_step, assistant_step and reset succeeds."""
    MinimalEngine()
    return True


def empty_init():
    """Returns True if trying to initialize an Interactionobseng
    without any arguments fails."""
    try:
        BaseObservationEngine()
        return True
    except TypeError:
        return False


def test_reset():
    obseng = BaseObservationEngine()
    obseng.reset()
    return True


def test_observe():
    obseng = BaseObservationEngine()
    with pytest.raises(AttributeError):
        obseng.observe()
    from coopihc.base.utils import example_state

    _example_state = example_state()
    obs = obseng.observe(_example_state)
    # Check equality on repr --- imperfect
    assert obs[0].__repr__() == _example_state.__repr__()


def test_observationengine():
    """Tests the methods provided by the Interactionobseng class."""
    test_imports()
    test_example()
    test_init()
    test_reset()
    test_observe()


# +----------------------+
# +        MAIN          +
# +----------------------+
if __name__ == "__main__":
    test_observationengine()
