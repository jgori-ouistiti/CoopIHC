from coopihc import InteractionTask


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


def can_be_subclassed_with_minimal_overrides():
    """Returns True if trying to subclass an InteractionTask with
    only overrides for user_step, assistant_step and reset succeeds."""

    class MinimalTask(InteractionTask):
        """Non-functional minimal subclass to use in tests."""

        def user_step(self):
            pass

        def assistant_step(self):
            pass

        def reset(self):
            pass

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


def test_interactiontask():
    """Tests the methods provided by the InteractionTask class."""
    test_imports()
    test_example()
    test_init()


# +----------------------+
# +        MAIN          +
# +----------------------+
if __name__ == "__main__":
    test_interactiontask()
