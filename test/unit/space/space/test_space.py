from coopihc.space.Space import Space
from coopihc.space.utils import discrete_space, continuous_space, multidiscrete_space
from coopihc.helpers import flatten
import numpy
import json


def test_init_discrete():
    # Discrete
    s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
    # prop and attributes
    assert s.dtype == numpy.int16
    assert s.space_type == "discrete"
    assert s.N == 3
    assert s.high == 3
    assert s.low == 1
    assert s.shape == (1,)
    # __ methods
    # __len__
    assert len(s) == 1


def test_contains_soft_discrete():
    s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete", contains="soft")
    assert 1 in s
    assert [1] in s
    assert [[1]] in s
    assert numpy.array(1) in s
    assert numpy.array([1]) in s
    assert numpy.array([[1]]) in s

    assert numpy.array([2]) in s
    assert numpy.array([3]) in s

    assert numpy.array([1.0]) not in s
    assert numpy.array([2], dtype=numpy.float32) not in s


def test_contains_hard_discrete():
    s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete", contains="hard")
    assert 1 not in s
    assert [1] not in s
    assert [[1]] not in s
    assert numpy.array(1) not in s
    assert numpy.array([1]) in s
    assert numpy.array([[1]]) not in s

    assert numpy.array([2]) in s
    assert numpy.array([3]) in s

    assert numpy.array([2.0]) not in s
    assert numpy.array([2], dtype=numpy.float32) not in s


def test_discrete():
    test_init_discrete()
    test_contains_soft_discrete()
    test_contains_hard_discrete()


def test_init_multidiscrete():
    # Discrete
    s = Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )
    # prop and attributes
    assert s.dtype == numpy.int16
    assert s.space_type == "multidiscrete"
    assert s.N == [3, 5]
    assert (s.high == numpy.array([3, 5])).all()
    assert (s.low == numpy.array([1, 1])).all()
    assert s.shape == (2, 1)
    # __ methods
    # __len__
    assert len(s) == 2


def test_contains_soft_multidiscrete():
    s = Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )
    assert [1, 1] in s
    assert [[1], [1]] in s
    assert [[1, 1]] in s
    assert numpy.array([1, 1]) in s
    assert numpy.array([[1], [1]]) in s
    assert numpy.array([[1, 1]]) in s

    assert numpy.array([[2], [5]]) in s
    assert numpy.array([[3], [3]]) in s

    assert numpy.array([[2.0], [5.0]]) not in s
    assert numpy.array([[2, 5]], dtype=numpy.float32) not in s
    assert numpy.array([[2], [5.0]]) not in s


def test_contains_hard_multidiscrete():
    s = Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="hard",
    )
    assert [1, 1] not in s
    assert [[1], [1]] not in s
    assert [[1, 1]] not in s
    assert numpy.array([1, 1]) not in s
    assert numpy.array([[1], [1]]) in s
    assert numpy.array([[1, 1]]) not in s

    assert numpy.array([[2], [5]]) in s
    assert numpy.array([[3], [3]]) in s

    assert numpy.array([[2.0], [5.0]]) not in s
    assert numpy.array([[2, 5]], dtype=numpy.float32) not in s
    assert numpy.array([[2], [5.0]]) not in s


def test_multidiscrete():
    test_init_multidiscrete()
    test_contains_soft_multidiscrete()
    test_contains_hard_multidiscrete()


def test_init_continuous():
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="hard",
    )
    # prop and attributes
    assert s.dtype == numpy.float32
    assert s.space_type == "continuous"
    assert s.N == None
    assert (s.high == numpy.ones((2, 2), dtype=numpy.float32)).all()
    assert (s.low == -numpy.ones((2, 2), dtype=numpy.float32)).all()
    assert s.shape == (2, 2)
    assert len(s) == 4


def test_contains_soft_continuous():
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="soft",
    )
    assert [0.0, 0.0, 0.0, 0.0] in s
    assert [[0.0, 0.0], [0.0, 0.0]] in s
    assert numpy.array([0.0, 0.0, 0.0, 0.0]) in s
    assert numpy.array([[0.0, 0.0], [0.0, 0.0]]) in s

    assert 1.0 * numpy.ones((2, 2)) in s
    assert -1.0 * numpy.ones((2, 2)) in s

    assert numpy.ones((2, 2), dtype=numpy.int16) in s


def test_contains_hard_continuous():
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="hard",
    )
    assert [0.0, 0.0, 0.0, 0.0] not in s
    assert [[0.0, 0.0], [0.0, 0.0]] not in s
    assert numpy.array([0.0, 0.0, 0.0, 0.0]) not in s
    assert numpy.array([[0.0, 0.0], [0.0, 0.0]]) in s

    assert 1.0 * numpy.ones((2, 2)) in s
    assert -1.0 * numpy.ones((2, 2)) in s

    assert numpy.ones((2, 2), dtype=numpy.int16) in s


def test_continuous():
    test_init_continuous()
    test_contains_soft_continuous()
    test_contains_hard_continuous()


def test_dtype_discrete():
    s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
    assert s.dtype == numpy.int16
    s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete", dtype=numpy.int64)
    assert s.dtype == numpy.int64


def test_dtype_continuous():
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
    )
    assert s.dtype == numpy.float32
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float64),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
    )
    assert s.dtype == numpy.float64
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        dtype=numpy.float64,
    )
    assert s.dtype == numpy.float64
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float64),
            numpy.ones((2, 2), dtype=numpy.float64),
        ],
        "continuous",
        dtype=numpy.int16,
    )
    assert s.dtype == numpy.int16


def test_dtype_multidiscrete():
    s = Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="hard",
    )
    assert s.dtype == numpy.int16
    s = Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int64),
        ],
        "multidiscrete",
        contains="hard",
    )
    assert s.dtype == numpy.int64
    s = Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="hard",
        dtype=numpy.int64,
    )
    assert s.dtype == numpy.int64


def test_dtype():
    test_dtype_discrete()
    test_dtype_continuous()
    test_dtype_multidiscrete()


def test_equal_discrete():
    s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
    assert s == Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
    assert s == numpy.array([1, 2, 3], dtype=numpy.int16)
    assert s != Space(numpy.array([1, 2, 3, 4], dtype=numpy.int16), "discrete")
    assert s == numpy.array([[1, 2, 3]], dtype=numpy.int16)
    assert s == numpy.array([[1, 2, 3]], dtype=numpy.float32)


def test_equal_multidiscrete():
    s = Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )
    assert s == Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )
    assert s == [
        numpy.array([1, 2, 3], dtype=numpy.int16),
        numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
    ]
    assert s != [numpy.array([1, 2, 3], dtype=numpy.int16)]
    assert s != Space(
        [
            numpy.array([1, 2, 3, 5], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )
    assert s != Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )


def test_equal_continuous():
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="soft",
    )
    assert s == Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="soft",
    )
    assert s == [
        -numpy.ones((2, 2), dtype=numpy.float32),
        numpy.ones((2, 2), dtype=numpy.float32),
    ]
    assert s != [
        -1.05 * numpy.ones((2, 2), dtype=numpy.float32),
        numpy.ones((2, 2), dtype=numpy.float32),
    ]
    assert s != Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            1.05 * numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="soft",
    )
    assert s != Space(
        [
            -numpy.ones((3, 2), dtype=numpy.float32),
            numpy.ones((3, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="soft",
    )


def test_equal():
    test_equal_discrete()
    test_equal_multidiscrete()
    test_equal_continuous()


def test_iter_discrete():
    s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
    assert iter(s).__next__() == s


def test_iter_continuous():
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="soft",
    )
    for _space in s:
        assert _space == Space(
            [
                -numpy.ones((1, 2), dtype=numpy.float32),
                numpy.ones((1, 2), dtype=numpy.float32),
            ],
            "continuous",
            contains="soft",
        )
        for __space in _space:
            assert __space == Space(
                [
                    -numpy.ones((1, 1), dtype=numpy.float32),
                    numpy.ones((1, 1), dtype=numpy.float32),
                ],
                "continuous",
                contains="soft",
            )


def test_iter_multidiscrete():
    s = Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )
    iterator = iter(s)
    assert iterator.__next__() == Space(
        numpy.array([1, 2, 3], dtype=numpy.int16), "discrete", contains="soft"
    )
    assert s.n == 1
    assert iterator.__next__() == Space(
        numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16), "discrete", contains="soft"
    )


def test_iter():
    test_iter_discrete()
    test_iter_continuous()
    test_iter_multidiscrete()


def test_sample_discrete():
    s = Space(
        numpy.array([i for i in range(1000)], dtype=numpy.int16),
        "discrete",
        contains="hard",
        seed=123,
    )
    q = Space(
        numpy.array([i for i in range(1000)], dtype=numpy.int16),
        "discrete",
        contains="soft",
        seed=123,
    )
    r = Space(
        numpy.array([i for i in range(1000)], dtype=numpy.int16),
        "discrete",
        contains="hard",
        seed=12,
    )
    _s, _q, _r = s.sample(), q.sample(), r.sample()
    assert _s == _q
    assert _s != _r

    s = Space(
        numpy.array([1, 2, 3], dtype=numpy.int16),
        "discrete",
        contains="hard",
        seed=123,
    )
    scont = {}
    for i in range(1000):
        _s = s.sample()
        scont.update({str(_s): _s})
        assert _s in s
    assert sorted(scont.values()) == [1, 2, 3]


def test_sample_continuous():
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="hard",
        seed=456,
    )
    q = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="soft",
        seed=456,
    )
    r = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="hard",
        seed=111,
    )

    _s, _q, _r = s.sample(), q.sample(), r.sample()
    assert (_s == _q).all()
    assert (_s != _r).any()
    for i in range(1000):
        assert s.sample() in s


def test_sample_multidiscrete():
    s = Space(
        [
            numpy.array([i for i in range(100)], dtype=numpy.int16),
            numpy.array([i for i in range(100)], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="hard",
        seed=789,
    )
    q = Space(
        [
            numpy.array([i for i in range(100)], dtype=numpy.int16),
            numpy.array([i for i in range(100)], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
        seed=789,
    )
    r = Space(
        [
            numpy.array([i for i in range(100)], dtype=numpy.int16),
            numpy.array([i for i in range(100)], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="hard",
        seed=123,
    )
    _s, _q, _r = s.sample(), q.sample(), r.sample()
    assert (_s == _q).all()
    assert (_s != _r).any()
    for i in range(1000):
        assert s.sample() in s


def test_sample():
    test_sample_discrete()
    test_sample_continuous()
    test_sample_multidiscrete()


def test_serialize_discrete():
    s = Space(
        numpy.array([i for i in range(10)], dtype=numpy.int16),
        "discrete",
        contains="hard",
        seed=123,
    )
    assert (
        json.dumps(s.serialize())
        == '{"array_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "space_type": "discrete", "seed": 123, "contains": "hard"}'
    )


def test_serialize_continuous():
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="hard",
        seed=456,
    )
    assert (
        json.dumps(s.serialize())
        == '{"array_list": [[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]], "space_type": "continuous", "seed": 456, "contains": "hard"}'
    )


def test_serialize_multidiscrete():
    s = Space(
        [
            numpy.array([i for i in range(10)], dtype=numpy.int16),
            numpy.array([i for i in range(10)], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="hard",
        seed=789,
    )
    assert (
        json.dumps(s.serialize())
        == '{"array_list": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], "space_type": "multidiscrete", "seed": 789, "contains": "hard"}'
    )


def test_serialize():
    test_serialize_discrete()
    test_serialize_continuous()
    test_serialize_multidiscrete()


if __name__ == "__main__":
    test_discrete()
    test_multidiscrete()
    test_continuous()
    test_dtype()
    test_equal()
    test_iter()
    test_sample()
    test_serialize()
