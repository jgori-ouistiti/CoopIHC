from coopihc.base.Space import Numeric, CatSet
from coopihc.base.utils import cartesian_product, space, SpaceNotSeparableError
from coopihc.helpers import flatten
import numpy
import json
import pytest


def test_init_CatSet():
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
    # prop and attributes
    assert s.dtype == numpy.int16
    assert s.N == 3
    assert s.shape == ()


def test_contains_CatSet():
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
    assert 1 in s
    assert [1] in s
    assert [[1]] in s
    assert numpy.array(1) in s
    assert numpy.array([1]) in s
    assert numpy.array([[1]]) in s

    assert numpy.array([2]) in s
    assert numpy.array([3]) in s

    assert numpy.array([1.0]) in s
    assert numpy.array([2]) in s


def test_CatSet():
    test_init_CatSet()
    test_contains_CatSet()


def test_init_Numeric():
    s = space(
        low=-numpy.ones((2, 2), dtype=numpy.float32),
        high=numpy.ones((2, 2), dtype=numpy.float32),
    )
    # prop and attributes
    assert s.dtype == numpy.float32
    assert (s.high == numpy.ones((2, 2))).all()
    assert (s.low == -numpy.ones((2, 2))).all()
    assert s.shape == (2, 2)

    s = space(low=-numpy.float64(1), high=numpy.float64(1))


def test_contains_Numeric():
    s = space(
        low=-numpy.ones((2, 2)),
        high=numpy.ones((2, 2)),
    )

    assert [0.0, 0.0, 0.0, 0.0] not in s
    assert [[0.0, 0.0], [0.0, 0.0]] in s
    assert numpy.array([0.0, 0.0, 0.0, 0.0]) not in s
    assert numpy.array([[0.0, 0.0], [0.0, 0.0]]) in s

    assert 1.0 * numpy.ones((2, 2)) in s
    assert -1.0 * numpy.ones((2, 2)) in s

    assert numpy.ones((2, 2), dtype=numpy.int16) in s


def test_Numeric():
    test_init_Numeric()
    test_contains_Numeric()


def test_sample_CatSet():
    s = space(array=numpy.arange(1000), seed=123)
    q = space(array=numpy.arange(1000), seed=123)
    r = space(array=numpy.arange(1000), seed=12)
    _s, _q, _r = s.sample(), q.sample(), r.sample()
    assert _s in s
    assert _q in q
    assert _r in r
    assert _s == _q
    assert _s != _r

    s = space(array=numpy.arange(4), seed=123)
    scont = {}
    for i in range(1000):
        _s = s.sample()
        scont.update({str(_s): _s})
        assert _s in s
    assert sorted(scont.values()) == [0, 1, 2, 3]


def test_sample_Numeric():
    s = space(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)), seed=123)
    q = space(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)), seed=123)
    r = space(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)), seed=12)

    _s, _q, _r = s.sample(), q.sample(), r.sample()
    assert _s in s
    assert _q in q
    assert _r in r
    assert (_s == _q).all()
    assert (_s != _r).any()
    for i in range(1000):
        assert s.sample() in s


def test_sample():
    test_sample_CatSet()
    test_sample_Numeric()


def test_dtype_CatSet():
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
    assert s.dtype == numpy.int16
    assert s.sample().dtype == numpy.int16
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16), dtype=numpy.int64)
    assert s.dtype == numpy.int64
    assert s.sample().dtype == numpy.int64


def test_dtype_Numeric():
    s = space(
        low=-numpy.ones((2, 2)),
        high=numpy.ones((2, 2)),
    )
    assert s.dtype == numpy.float64
    assert s.sample().dtype == numpy.float64

    s = space(
        low=-numpy.ones((2, 2), dtype=numpy.float32),
        high=numpy.ones((2, 2), dtype=numpy.float32),
    )
    assert s.dtype == numpy.float32
    assert s.sample().dtype == numpy.float32

    s = space(
        low=-numpy.ones((2, 2)),
        high=numpy.ones((2, 2)),
        dtype=numpy.float32,
    )
    assert s.dtype == numpy.float32
    assert s.sample().dtype == numpy.float32
    s = space(
        low=-numpy.ones((2, 2)),
        high=numpy.ones((2, 2)),
        dtype=numpy.int16,
    )
    assert s.dtype == numpy.int16
    assert s.sample().dtype == numpy.int16


def test_dtype():
    test_dtype_CatSet()
    test_dtype_Numeric()


def test_equal_CatSet():
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
    assert s == space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
    assert s != space(array=numpy.array([1, 2, 3, 4], dtype=numpy.int16))


def test_equal_Numeric():
    s = space(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)))
    assert s == space(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)))
    assert s != space(low=-1.5 * numpy.ones((2, 2)), high=2 * numpy.ones((2, 2)))
    assert s != space(low=-numpy.ones((1,)), high=numpy.ones((1,)))


def test_equal():
    test_equal_CatSet()
    test_equal_Numeric()


def test_serialize_CatSet():
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
    assert (
        json.dumps(s.serialize())
        == '{"space": "CatSet", "seed": null, "array": [1, 2, 3], "dtype": "dtype[int16]"}'
    )


def test_serialize_Numeric():
    s = space(
        low=-numpy.ones((2, 2)),
        high=numpy.ones((2, 2)),
    )
    assert (
        json.dumps(s.serialize())
        == '{"space": "Numeric", "seed": null, "low,high": [[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]], "shape": [2, 2], "dtype": "dtype[float64]"}'
    )


def test_serialize():
    test_serialize_CatSet()
    test_serialize_Numeric()


def test_iter_CatSet():
    pass


def test_iter_Numeric():
    s = space(low=numpy.array([[-1, -2], [-3, -4]]), high=numpy.array([[1, 2], [3, 4]]))
    for i, _s in enumerate(s):
        if i == 0:
            assert _s == space(low=numpy.array([-1, -2]), high=-numpy.array([-1, -2]))
        if i == 1:
            assert _s == space(low=numpy.array([-3, -4]), high=-numpy.array([-3, -4]))
        for j, _ss in enumerate(_s):
            if i == 0 and j == 0:
                assert _ss == space(low=-numpy.int64(1), high=numpy.int64(1))
            elif i == 0 and j == 1:
                assert _ss == space(low=-numpy.int64(2), high=numpy.int64(2))
            elif i == 1 and j == 0:
                assert _ss == space(low=-numpy.int64(3), high=numpy.int64(3))
            elif i == 1 and j == 1:
                assert _ss == space(low=-numpy.int64(4), high=numpy.int64(4))


def test_iter():
    test_iter_CatSet()
    test_iter_Numeric()


def test_cartesian_product_CatSet():
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
    q = space(array=numpy.array([-3, -2, -1], dtype=numpy.int16))
    cp, _ = cartesian_product(s, q)
    assert (
        cp
        == numpy.array(
            [
                [1, -3],
                [1, -2],
                [1, -1],
                [2, -3],
                [2, -2],
                [2, -1],
                [3, -3],
                [3, -2],
                [3, -1],
            ]
        )
    ).all()


def test_cartesian_product_Numeric():
    s = space(
        low=-numpy.ones((2, 2)),
        high=numpy.ones((2, 2)),
    )
    q = space(
        low=-numpy.ones((2, 2)),
        high=numpy.ones((2, 2)),
    )
    cp, _shape = cartesian_product(s, q)
    assert (cp == numpy.array([[None, None]])).all()


def test_cartesian_product_mix():
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
    q = Numeric(
        low=-numpy.ones((2, 2)),
        high=numpy.ones((2, 2)),
    )
    r = space(array=numpy.array([5, 6, 7], dtype=numpy.int16))
    cp, shape = cartesian_product(s, q, r)
    assert (
        cp
        == numpy.array(
            [
                [1, None, 5],
                [1, None, 6],
                [1, None, 7],
                [2, None, 5],
                [2, None, 6],
                [2, None, 7],
                [3, None, 5],
                [3, None, 6],
                [3, None, 7],
            ]
        )
    ).all()
    assert shape == [(), (2, 2), ()]


def test_cartesian_product_single():
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
    cp, shape = cartesian_product(s)
    assert (cp == numpy.array([[1], [2], [3]])).all()
    assert shape == [()]


def test_cartesian_product():
    test_cartesian_product_CatSet()
    test_cartesian_product_Numeric()
    test_cartesian_product_mix()
    test_cartesian_product_single()


def test__getitem__CatSet():
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
    with pytest.raises(SpaceNotSeparableError):
        s[0]
    assert s[...] == s
    assert s[:] == s


def test__getitem__int_interval():
    s = space(
        low=-numpy.ones((2, 2), dtype=numpy.float32),
        high=numpy.ones((2, 2), dtype=numpy.float32),
    )
    assert s[0] == space(
        low=-numpy.ones((2,), dtype=numpy.float32),
        high=numpy.ones((2,), dtype=numpy.float32),
    )


def test__getitem__slice_interval():
    s = space(
        low=-numpy.ones((2, 2), dtype=numpy.float32),
        high=numpy.ones((2, 2), dtype=numpy.float32),
    )
    assert s[:, 0] == space(
        low=-numpy.ones((2,), dtype=numpy.float32),
        high=numpy.ones((2,), dtype=numpy.float32),
    )
    assert s[0, :] == space(
        low=-numpy.ones((2,), dtype=numpy.float32),
        high=numpy.ones((2,), dtype=numpy.float32),
    )
    assert s[:, :] == s
    assert s[...] == s


def test__getitem__():
    test__getitem__CatSet()
    test__getitem__int_interval()
    test__getitem__slice_interval()


def test_N_Numeric():
    s = Numeric(low=numpy.array(-2), high=numpy.array(3), dtype=numpy.int8)
    assert s.N == 6


def test_array_Numeric():
    s = Numeric(low=numpy.array(-2), high=numpy.array(3), dtype=numpy.int8)
    assert (s.array == numpy.array([-2, -1, 0, 1, 2, 3])).all()
    assert s.dtype == s.array.dtype


if __name__ == "__main__":
    test_CatSet()
    test_Numeric()
    test_sample()
    test_dtype()
    test_equal()
    test_serialize()
    test_iter()
    test_cartesian_product()
    test__getitem__()
    test_N_Numeric()
    test_array_Numeric()
