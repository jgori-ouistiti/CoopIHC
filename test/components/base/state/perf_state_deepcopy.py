import copy
from coopihc import State, StateElement, example_game_state


import cProfile


def test_state_deepcopy_round(benchmark):
    """Runs a performance test for 1000 deepcopies of example game state."""
    benchmark(state_deepcopy_round)


def state_deepcopy_round():
    """Runs a performance test for 1000 deepcopies of example game state."""
    eg = example_game_state()
    for i in range(1000):
        copy.deepcopy(eg)
    # Bundle a task together with two BaseAgents


# +----------------------+
# +        MAIN          +
# +----------------------+
if __name__ == "__main__":
    cProfile.run("state_deepcopy_round()", sort="cumulative")
