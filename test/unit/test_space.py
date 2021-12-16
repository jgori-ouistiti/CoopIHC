from coopihc.space.Space import Space
from coopihc.helpers import flatten
import numpy
import copy


def test_init():
    # ===================== 1D int
    s = Space([numpy.array([1, 2, 3], dtype=numpy.int16)])
    # prop and attributes
    assert s.dtype == numpy.int16
    assert s.continuous == False
    assert s.shape == (1, 1)
    assert s.N == 3
    assert s.high == [3]
    assert s.low == [1]
    assert (s.range == numpy.atleast_2d(numpy.array([1, 2, 3]))).all()
    # __ methods
    # __len__
    assert len(s) == 1
    # __contains__
    assert numpy.array([1]) in s
    assert numpy.array([2]) in s
    assert numpy.array([3]) in s
    assert numpy.array([[2]]).reshape(1, -1) in s
    assert numpy.array([[2]]).reshape(-1, 1) in s
    assert numpy.array([2.0]) not in s
    assert numpy.array([2], dtype=numpy.float32) not in s
    # __iter__ and __eq__  ---> here iter quasi-idempotent (== object, but not identical)
    for _s in s:
        assert _s == s
    q = Space([numpy.array([1, 1, 3], dtype=numpy.int16)])
    assert q != s
    r = Space([numpy.array([1, 2, 3], dtype=numpy.float32)])
    assert s != r

    # ========================= single 2D float
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ]
    )
    # prop and attributes
    assert s.dtype == numpy.float32
    assert s.continuous == True
    assert s.shape == (2, 2)
    assert s.N == None
    assert (s.high == numpy.ones((2, 2), numpy.float32)).all()
    assert (s.low == -numpy.ones((2, 2), numpy.float32)).all()
    assert (s.range[0] == -numpy.ones((2, 2), dtype=numpy.float32)).all()
    assert (s.range[1] == numpy.ones((2, 2), dtype=numpy.float32)).all()

    # __ methods
    # __len__
    assert len(s) == 2
    # __contains__
    assert -1.0 * numpy.eye(2, 2) in s
    assert 1.0 * numpy.eye(2, 2) in s
    assert 0 * numpy.eye(2, 2) in s
    assert 1 * numpy.eye(2, 2) in s
    assert -1 * numpy.eye(2, 2) in s
    assert 2 * numpy.eye(2, 2) not in s
    # __eq__
    ss = Space(
        [
            -1.0 * numpy.ones((2, 2), dtype=numpy.float32),
            1.0 * numpy.ones((2, 2), dtype=numpy.float32),
        ]
    )
    assert ss == s
    sss = Space(
        [
            -1.0 * numpy.ones((2, 2)),
            1.0 * numpy.ones((2, 2)),
        ]
    )
    assert sss != s
    q = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.int16),
            numpy.ones((2, 2), dtype=numpy.int16),
        ]
    )
    r = Space(
        [
            -2 * numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ]
    )
    assert q != s
    assert r != s

    # __iter__
    for _s in s:
        assert 0.5 * numpy.eye(1, 2) in _s
        for _ss in _s:
            assert numpy.array([[0.5]]) in _ss

    # ====================== multi 1D int
    gridsize = (31, 31)
    s = Space(
        [
            numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
            numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
        ]
    )
    # prop and attributes
    assert s.dtype == numpy.int16
    assert s.continuous == False
    assert s.shape == (2, 1)
    assert s.N == None
    assert s.high == [30, 30]
    assert s.low == [0, 0]
    assert (s.range[0] == numpy.array([[i for i in range(31)]])).all()
    assert (s.range[1] == numpy.array([[i for i in range(31)]])).all()

    # __ methods
    # __len__
    assert len(s) == 2
    # __contains__
    assert numpy.array([1, 2]) in s
    assert numpy.array([-2, 5]) not in s
    assert numpy.array([1, 35]) not in s

    # __eq__
    ss = Space(
        [
            numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
            numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
        ]
    )
    assert ss == s
    sss = Space(
        [
            numpy.array([i for i in range(29)], dtype=numpy.int16),
            numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
        ]
    )
    assert sss != s
    ssss = Space(
        [
            numpy.array([i for i in range(31)], dtype=numpy.int16),
            numpy.array([i for i in range(5)], dtype=numpy.int16),
        ]
    )
    assert ssss != s
    q = Space(
        [
            numpy.array([i - 4 for i in range(31)], dtype=numpy.int16),
            numpy.array([i for i in range(31)], dtype=numpy.int16),
        ]
    )
    r = Space(
        [
            numpy.array([i for i in range(31)], dtype=numpy.int16),
            numpy.array([i + 1 for i in range(31)], dtype=numpy.int16),
        ]
    )
    assert q != s
    assert r != s

    # __iter__
    for _s in s:
        assert _s == Space(
            [numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16)]
        )

    # ========= multi int 2D
    number_of_targets = 3
    s = Space(
        flatten(
            [
                [
                    numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
                    numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
                ]
                for j in range(number_of_targets)
            ]
        )
    )

    # prop and attributes
    assert s.dtype == numpy.int16
    assert s.continuous == False
    assert s.shape == (6, 1)
    assert s.N == None
    assert s.high == [30, 30, 30, 30, 30, 30]
    assert s.low == [0, 0, 0, 0, 0, 0]
    for i in range(s.shape[0]):
        assert (s.range[i] == numpy.array([[i for i in range(31)]])).all()

    # __ methods
    # __len__
    assert len(s) == 6
    # __contains__
    assert numpy.array([1, 2, 4, 5, 3, 2]) in s
    assert numpy.array([-2, 5, 1, 1, 1, 1]) not in s
    assert numpy.array([1, 35, 1, 1, 1, 1]) not in s
    assert numpy.array([1, 35, 1, 1]) not in s

    # __eq__

    ss = Space(
        [numpy.array([i for i in range(31)], dtype=numpy.int16) for j in range(6)]
    )
    assert ss == s

    sss = Space(
        [
            numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
            numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
        ]
    )
    assert sss != s
    ssss = Space(
        flatten(
            [
                [
                    numpy.array([i for i in range(31)], dtype=numpy.int16)
                    for j in range(5)
                ],
                [numpy.array([i for i in range(5)], dtype=numpy.int16)],
            ]
        )
    )
    assert ssss != s

    q = Space(
        [numpy.array([i - j for i in range(31)], dtype=numpy.int16) for j in range(6)]
    )
    r = Space(
        [numpy.array([i + j for i in range(31)], dtype=numpy.int16) for j in range(6)]
    )
    assert q != s
    assert r != s

    # __iter__
    for n, _s in enumerate(s):
        assert _s == Space(
            [numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16)]
        )
    assert n == 5

    # =========== None single space

    s = Space([numpy.array([None], dtype=object)])
    # prop and attributes
    assert s.dtype == object
    assert s.continuous == False
    assert s.shape == (1, 1)
    assert s.range == [None]
    assert s.high == [None]
    assert s.low == [None]
    assert s.N == None

    # __ methods
    # __len__
    assert len(s) == 1


def test_sample():
    # ================== Discrete
    number_of_targets = 3
    gridsize = [5, 5]
    s = Space(
        flatten(
            [
                [
                    numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
                    numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
                ]
                for j in range(number_of_targets)
            ]
        ),
        seed=123,
    )

    # -------- Check if samples are valid
    for i in range(100):
        assert s.sample() in s
    # --------- Check two samples are different
    assert (s.sample() != s.sample()).any()
    # --------- Check that seeding works
    ss = copy.deepcopy(s)
    assert (ss.sample() == s.sample()).all()
    sss = Space(
        flatten(
            [
                [
                    numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
                    numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
                ]
                for j in range(number_of_targets)
            ]
        ),
        seed=123,
    )
    ssss = Space(
        flatten(
            [
                [
                    numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
                    numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
                ]
                for j in range(number_of_targets)
            ]
        ),
        seed=123,
    )

    sfive = Space(
        flatten(
            [
                [
                    numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
                    numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
                ]
                for j in range(number_of_targets)
            ]
        ),
        seed=13,
    )

    a1 = sss.sample()
    a2 = ssss.sample()
    a3 = sfive.sample()
    assert (a1 == a2).all()
    assert (a1 != a3).any()
    # =============== Continuous
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ]
    )
    # -------- Check if samples are valid
    for i in range(100):
        assert s.sample() in s
    # --------- Check two samples are different
    assert (s.sample() != s.sample()).any()
    # --------- Check that seeding works
    ss = copy.deepcopy(s)
    assert (ss.sample() == s.sample()).all()


def test_eq():
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ]
    )
    v = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ]
    )
    w = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float64),
            numpy.ones((2, 2), dtype=numpy.float64),
        ]
    )
    assert s == v
    assert s != w


# +----------------------+
# +        MAIN          +
# +----------------------+
if __name__ == "__main__":
    test_init()
    test_sample()
    test_eq()
