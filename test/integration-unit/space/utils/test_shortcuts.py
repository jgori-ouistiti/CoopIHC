import numpy
from coopihc.space.utils import lin_space, box_space, integer_space


def test_lin_space():
    space = lin_space(10, dtype=numpy.int16)
    space = lin_space(num=11, start=-5)


def test_integer():
    space = integer_space(10, dtype=numpy.int16)


def test_box():
    space = box_space(numpy.ones((3, 3)))
    space = box_space(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)))


def test_base_init():
    test_lin_space()
    test_box()


if __name__ == "__main__":
    test_base_init()
