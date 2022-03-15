from coopihc.space.Space import Interval, CatSet, space
from coopihc.space.Space import cartesian_product
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


def test_init_Interval():
    s = space(
        low=-numpy.ones((2, 2), dtype=numpy.float32),
        high=numpy.ones((2, 2), dtype=numpy.float32),
    )
    # prop and attributes
    assert s.dtype == numpy.float32
    assert (s.high == numpy.ones((2, 2))).all()
    assert (s.low == -numpy.ones((2, 2))).all()
    assert s.shape == (2, 2)


def test_contains_Interval():
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


def test_Interval():
    test_init_Interval()
    test_contains_Interval()


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


def test_sample_Interval():
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
    test_sample_Interval()


def test_dtype_CatSet():
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
    assert s.dtype == numpy.int16
    assert s.sample().dtype == numpy.int16
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16), dtype=numpy.int64)
    assert s.dtype == numpy.int64
    assert s.sample().dtype == numpy.int64


def test_dtype_Interval():
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
    test_dtype_Interval()


# def test_equal_CatSet():
#     s = space(numpy.array([1, 2, 3], dtype=numpy.int16), "CatSet")
#     assert s == space(numpy.array([1, 2, 3], dtype=numpy.int16), "CatSet")
#     assert s == numpy.array([1, 2, 3], dtype=numpy.int16)
#     assert s != space(numpy.array([1, 2, 3, 4], dtype=numpy.int16), "CatSet")
#     assert s == numpy.array([[1, 2, 3]], dtype=numpy.int16)
#     assert s == numpy.array([[1, 2, 3]])


# def test_equal_multiCatSet():
#     s = space(
#         [
#             numpy.array([1, 2, 3], dtype=numpy.int16),
#             numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
#         ],
#         "multiCatSet",
#         contains="soft",
#     )
#     assert s == space(
#         [
#             numpy.array([1, 2, 3], dtype=numpy.int16),
#             numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
#         ],
#         "multiCatSet",
#         contains="soft",
#     )
#     assert s == [
#         numpy.array([1, 2, 3], dtype=numpy.int16),
#         numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
#     ]
#     assert s != [numpy.array([1, 2, 3], dtype=numpy.int16)]
#     assert s != space(
#         [
#             numpy.array([1, 2, 3, 5], dtype=numpy.int16),
#             numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
#         ],
#         "multiCatSet",
#         contains="soft",
#     )
#     assert s != space(
#         [
#             numpy.array([1, 2, 3], dtype=numpy.int16),
#             numpy.array([1, 2, 3, 5], dtype=numpy.int16),
#         ],
#         "multiCatSet",
#         contains="soft",
#     )


# def test_equal_Interval():
#     s = space(
#         [
#             -numpy.ones((2, 2)),
#             numpy.ones((2, 2)),
#         ],
#         "Interval",
#         contains="soft",
#     )
#     assert s == space(
#         [
#             -numpy.ones((2, 2)),
#             numpy.ones((2, 2)),
#         ],
#         "Interval",
#         contains="soft",
#     )
#     assert s == [
#         -numpy.ones((2, 2)),
#         numpy.ones((2, 2)),
#     ]
#     assert s != [
#         -1.05 * numpy.ones((2, 2)),
#         numpy.ones((2, 2)),
#     ]
#     assert s != space(
#         [
#             -numpy.ones((2, 2)),
#             1.05 * numpy.ones((2, 2)),
#         ],
#         "Interval",
#         contains="soft",
#     )
#     assert s != space(
#         [
#             -numpy.ones((3, 2)),
#             numpy.ones((3, 2)),
#         ],
#         "Interval",
#         contains="soft",
#     )


# def test_equal():
#     test_equal_CatSet()
#     test_equal_multiCatSet()
#     test_equal_Interval()


def test_serialize_CatSet():
    s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
    assert (
        json.dumps(s.serialize())
        == '{"space": "CatSet", "seed": null, "array": [1, 2, 3], "dtype": "dtype[int16]"}'
    )


def test_serialize_Interval():
    s = space(
        low=-numpy.ones((2, 2)),
        high=numpy.ones((2, 2)),
    )
    assert (
        json.dumps(s.serialize())
        == '{"space": "Interval", "seed": null, "low,high": [[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]], "shape": [2, 2], "dtype": "type"}'
    )


def test_serialize():
    test_serialize_CatSet()
    test_serialize_Interval()


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


def test_cartesian_product_Interval():
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
    q = Interval(
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
    test_cartesian_product_Interval()
    test_cartesian_product_mix()
    test_cartesian_product_single()


if __name__ == "__main__":
    test_CatSet()
    test_Interval()
    test_sample()
    test_dtype()
    # test_equal()
    test_serialize()
    test_cartesian_product()
