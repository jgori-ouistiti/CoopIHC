import numpy

# from coopihc.base.elements import lin_space
from coopihc.base.elements import box_space, integer_space, integer_set
from coopihc.base.Space import CatSet, Numeric


def test_lin_space():
    # space = lin_space(0, 10, num=11, dtype=numpy.int16)
    # assert space.dtype == numpy.int16
    # assert space.low == 0
    # assert space.high == 10
    # assert space.N == 11
    # space = lin_space(-5, 5, num=11)
    # assert space.dtype == numpy.int64
    # assert space.low == -5
    # assert space.high == 5
    # assert space.N == 11
    pass


def test_integer_space():
    space = integer_space(10, dtype=numpy.int16)
    assert isinstance(space, Numeric)
    assert space.dtype == numpy.int16
    assert space.low == 0
    assert space.high == 9
    assert space.N == 10
    space = integer_space(N=3, start=-1)
    assert isinstance(space, Numeric)
    assert space.dtype == numpy.int64
    assert space.low == -1
    assert space.high == 1
    assert space.N == 3


def test_integer_set():
    space = integer_set(10, dtype=numpy.int16)
    assert isinstance(space, CatSet)
    assert space.dtype == numpy.int16
    assert space.low == 0


def test_box():
    space = box_space(numpy.ones((3, 3)))
    assert isinstance(space, Numeric)
    assert space.dtype == numpy.float64
    assert (space.low == numpy.full((3, 3), -1)).all()
    assert (space.high == numpy.full((3, 3), 1)).all()

    space = box_space(low=-2 * numpy.ones((2, 2)), high=numpy.ones((2, 2)))
    assert isinstance(space, Numeric)
    assert space.dtype == numpy.float64
    assert (space.low == numpy.full((2, 2), -2)).all()
    assert (space.high == numpy.full((2, 2), 1)).all()


def test_base_init():
    test_lin_space()
    test_integer_space()
    test_integer_set()
    test_box()


if __name__ == "__main__":
    test_base_init()
