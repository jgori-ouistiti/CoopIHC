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


def test_interactiontask():
    """Tests the methods provided by the InteractionTask class."""
    test_imports()
    test_example()


# +----------------------+
# +        MAIN          +
# +----------------------+
if __name__ == "__main__":
    test_interactiontask()
