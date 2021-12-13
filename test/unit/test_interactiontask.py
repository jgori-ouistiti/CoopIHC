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
