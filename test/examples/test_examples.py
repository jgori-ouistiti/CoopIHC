import pytest


def test_basic_examples():
    import coopihc.examples.basic_examples.space_examples
    import coopihc.examples.basic_examples.stateelement_examples
    import coopihc.examples.basic_examples.state_examples
    import coopihc.examples.basic_examples.observation_examples
    import coopihc.examples.basic_examples.policy_examples
    import coopihc.examples.basic_examples.agents_examples
    import coopihc.examples.basic_examples.interactiontask_examples


def test_simple_examples():
    import coopihc.examples.simple_examples.lqr_example
    import coopihc.examples.simple_examples.lqg_example
    import coopihc.examples.simple_examples.assistant_has_user_model


def test_poiting_examples():
    import coopihc.examples.simplepointing.scripts.basic_big


@pytest.mark.timeout(3)
def test_bundle_examples():
    import coopihc.examples.basic_examples.bundle_examples


def test_all_examples():
    test_basic_examples()
    test_simple_examples()
    test_bundle_examples()


if __name__ == "__main__":
    test_all_examples()
