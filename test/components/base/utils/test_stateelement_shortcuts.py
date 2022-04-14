import numpy
from coopihc.base.elements import array_element, discrete_array_element, cat_element
from coopihc.base.Space import Numeric, CatSet


def test_array_element():
    x = array_element(init=1)
    assert x == 1
    assert x.shape == ()
    assert isinstance(x.space, Numeric)
    assert x.space.low == -numpy.inf
    assert x.space.high == numpy.inf
    assert x.dtype == numpy.float64
    assert x.seed == None

    x = array_element(shape=(1, 1))
    assert x == 0
    assert x.shape == (1, 1)
    assert isinstance(x.space, Numeric)
    assert x.space.low == -numpy.inf
    assert x.space.high == numpy.inf
    assert x.dtype == numpy.float64
    assert x.seed == None

    x = array_element(init=1, seed=123, dtype=numpy.float32)
    assert x == 1
    assert x.shape == ()
    assert isinstance(x.space, Numeric)
    assert x.space.low == -numpy.inf
    assert x.space.high == numpy.inf
    assert x.dtype == numpy.float32
    assert x.seed == 123

    x = array_element(
        init=numpy.array(1),
        low=numpy.array(-2, dtype=numpy.float32),
        high=numpy.array(2, dtype=numpy.float32),
    )
    assert x == 1
    assert x.shape == ()
    assert isinstance(x.space, Numeric)
    assert x.space.low == -2
    assert x.space.high == 2
    assert x.dtype == numpy.float32

    x = array_element(
        init=numpy.array(1),
        low=numpy.array(-numpy.inf, dtype=numpy.float32),
        high=numpy.array(numpy.inf, dtype=numpy.float32),
    )
    assert x == 1
    assert x.shape == ()
    assert isinstance(x.space, Numeric)
    assert x.space.low == -numpy.inf
    assert x.space.high == numpy.inf
    assert x.dtype == numpy.float32

    x = array_element(init=1, dtype=numpy.float32)
    assert x == 1
    assert x.shape == ()
    assert isinstance(x.space, Numeric)
    assert x.space.low == -numpy.inf
    assert x.space.high == numpy.inf
    assert x.dtype == numpy.float32


def test_discrete_array_element():
    x = discrete_array_element(N=5, init=1)
    assert x == 1
    assert x.shape == ()
    assert isinstance(x.space, Numeric)
    assert x.space.low == 0
    assert x.space.high == 4
    assert x.dtype == numpy.int64
    assert x.seed == None

    x = discrete_array_element(N=5, init=1, shape=(2, 2), seed=123, dtype=numpy.int8)
    assert (x == 1).all()
    assert (x == numpy.ones((2, 2))).all()
    assert x.shape == (2, 2)
    assert isinstance(x.space, Numeric)
    assert (x.space.low == 0).all()
    assert (x.space.high == 4).all()
    assert x.dtype == numpy.int8
    assert x.seed == 123

    x = discrete_array_element(init=-4, low=-4, high=4)
    assert x == -4
    assert x.shape == ()
    assert isinstance(x.space, Numeric)
    assert x.space.low == -4
    assert x.space.high == 4
    assert x.dtype == numpy.int64
    assert x.seed == None

    x = discrete_array_element(low=0, high=31, shape=(8,))
    assert (x == 0).all()
    assert x.shape == (8,)
    assert isinstance(x.space, Numeric)
    assert (x.space.low == 0).all()
    assert (x.space.high == 31).all()
    assert x.dtype == numpy.int64
    assert x.seed == None

    x = discrete_array_element(low=0, high=numpy.inf)
    assert x == 0
    assert x.shape == ()
    assert isinstance(x.space, Numeric)
    assert x.space.low == 0
    assert x.dtype == numpy.int64
    assert x.space.high <= numpy.iinfo(x.dtype).max
    assert x.space.high >= numpy.iinfo(x.dtype).max
    assert x.seed == None

    x = discrete_array_element(
        low=numpy.array(-(2 ** 14), dtype=numpy.int16), high=numpy.inf, dtype=numpy.int8
    )
    assert x == 0
    assert x.shape == ()
    assert isinstance(x.space, Numeric)
    assert x.space.low == -(2 ** 7)
    assert x.dtype == numpy.int8
    assert x.space.high <= numpy.iinfo(x.dtype).max
    assert x.space.high >= numpy.iinfo(x.dtype).max
    assert x.seed == None


def test_cat_element():
    x = cat_element(5, init=0)
    assert x == 0
    assert x.shape == ()
    assert isinstance(x.space, CatSet)
    assert x.space.low == 0
    assert x.space.high == 4
    assert x.dtype == numpy.int64
    assert x.seed == None

    x = cat_element(5, init=0, seed=123, dtype=numpy.int8)
    assert x == 0
    assert x.shape == ()
    assert isinstance(x.space, CatSet)
    assert x.space.low == 0
    assert x.space.high == 4
    assert x.dtype == numpy.int8
    assert x.seed == 123


def test_init():
    test_array_element()
    test_discrete_array_element()


if __name__ == "__main__":
    test_init()
