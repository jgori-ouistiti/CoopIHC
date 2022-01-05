import pytest


def test_simple_examples():
    import coopihc.examples.simple_examples.space_examples
    import coopihc.examples.simple_examples.stateelement_examples
    import coopihc.examples.simple_examples.state_examples
    import coopihc.examples.simple_examples.observation_examples
    import coopihc.examples.simple_examples.policy_examples
    import coopihc.examples.simple_examples.agents_examples
    import coopihc.examples.simple_examples.interactiontask_examples


@pytest.mark.timeout(3)
def test_bundle_examples():
    import coopihc.examples.simple_examples.bundle_examples


def test_all_examples():
    test_simple_examples()
    test_bundle_examples()


if __name__ == "__main__":
    test_all_examples()
