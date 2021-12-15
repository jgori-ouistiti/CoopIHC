import numpy
from coopihc.space.Space import Space
from coopihc.space.utils import discrete_space, continuous_space, multidiscrete_space
import gym


def test_all_conversions():
    test_discrete()
    test_continuous()
    test_multidiscrete()


def test_discrete():
    s = discrete_space([1, 2, 3])
    assert s.convert_to_gym() == [gym.spaces.Discrete(3)]


def test_continuous():
    s = continuous_space(
        -numpy.ones((2, 2)),
        numpy.ones((2, 2)),
    )
    assert s.convert_to_gym() == [
        gym.spaces.Box(
            low=-numpy.ones((2, 2), dtype=numpy.float32),
            high=numpy.ones((2, 2), dtype=numpy.float32),
        )
    ]


def test_multidiscrete():
    s = multidiscrete_space([[1, 2, 3], (4, 5, 6)])
    assert s.convert_to_gym() == [gym.spaces.Discrete(3), gym.spaces.Discrete(3)]


if __name__ == "__main__":
    test_all_conversions()
