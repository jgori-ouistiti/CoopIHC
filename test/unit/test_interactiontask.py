from coopihc import InteractionTask


def test_imports():
    """Tests the different import ways for the InteractionTask."""
    from coopihc import InteractionTask
    from coopihc.interactiontask import InteractionTask
    from coopihc.interactiontask.InteractionTask import InteractionTask


def test_empty_init():
    """Tries to initialize an InteractionTask without arguments."""
    InteractionTask()


def test_init():
    """Tests the initializer of the InteractionTask class."""
    test_empty_init()


def test_interactiontask():
    """Tests the methods provided by the InteractionTask class."""
    test_imports()
    test_init()


# +----------------------+
# +        MAIN          +
# +----------------------+
if __name__ == "__main__":
    test_interactiontask()
