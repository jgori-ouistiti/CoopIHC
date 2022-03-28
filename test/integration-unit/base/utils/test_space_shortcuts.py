import numpy
from coopihc.base.elements import lin_space, box_space, integer_space, integer_set


def test_lin_space():
    space = lin_space(0, 10, num=11, dtype=numpy.int16)
    space = lin_space(-5, 5, num=11)


def test_integer_space():
    space = integer_space(10, dtype=numpy.int16)
    space = integer_space(N=3, start=-1)
    assert space.low == -1
    assert space.high == 1


def test_integer_set():
    space = integer_set(10, dtype=numpy.int16)


def test_box():
    space = box_space(numpy.ones((3, 3)))
    space = box_space(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)))


def test_base_init():
    test_lin_space()
    test_integer_space()
    test_integer_set()
    test_box()


if __name__ == "__main__":
    test_base_init()
