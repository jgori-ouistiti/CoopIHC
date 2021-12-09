from coopihc.space.Space import Space
from coopihc.helpers import flatten
import numpy

import sys


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

    # number_of_targets = 1


#     h = Space(
#         flatten(
#             [
#                 [
#                     numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
#                     numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
#                 ]
#                 for j in range(number_of_targets)
#             ]
#         )
#     )


# if _str == "action-space":
#     h = Space(
#         flatten(
#             [
#                 [
#                     numpy.array([i for i in range(4)], dtype=numpy.int16),
#                     numpy.array([i for i in range(4)], dtype=numpy.int16),
#                 ]
#                 for j in range(3)
#             ]
#         )
#     )
#     s = Space([numpy.array([1, 2, 3], dtype=numpy.int16)])
#     a = Space([numpy.array([None], dtype=numpy.object)])

# if _str == "contains" or _str == "all":
#     space = Space([numpy.array([1, 2, 3], dtype=numpy.int16)])
#     x = numpy.array([2], dtype=numpy.int16)
#     y = numpy.array([2], dtype=numpy.float32)
#     yy = numpy.array([2])
#     z = numpy.array([5])
#     assert x in space
#     assert y not in space
#     assert yy in space
#     assert z not in space

#     space = Space(
#         [-numpy.ones((2, 2), dtype=numpy.float32), numpy.ones((2, 2), numpy.float32)]
#     )
#     x = numpy.array([[1, 1], [1, 1]], dtype=numpy.int16)
#     y = numpy.array([[1, 1], [1, 1]], dtype=numpy.float32)
#     yy = numpy.array([[1, 1], [1, 1]])
#     yyy = numpy.array([[1.0, 1.0], [1.0, 1.0]])
#     z = numpy.array([[5, 1], [1, 1]], dtype=numpy.float32)
#     assert x in space
#     assert y in space
#     assert yy in space
#     assert yyy in space
#     assert z not in space

#     gridsize = (31, 31)
#     g = Space(
#         [
#             numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
#             numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
#         ]
#     )

#     x = numpy.array([2, 3])
#     xx = numpy.array([2, 3, 1])
#     xxx = numpy.array([2.0, 3.0])
#     xxxx = numpy.array([150, 21])

#     assert x in g
#     assert xx not in g
#     assert xxx not in g
#     assert xxxx not in g

# if _str == "sample" or _str == "all":
#     gridsize = (31, 31)
#     f = Space(
#         [
#             numpy.array([[-2, -2], [-1, -1]], dtype=numpy.float32),
#             numpy.ones((2, 2), numpy.float32),
#         ]
#     )
#     g = Space(
#         [
#             numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
#             numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
#         ]
#     )
#     h = Space([numpy.array([i for i in range(10)], dtype=numpy.int16)])

#     f.sample()
#     g.sample()
#     h.sample()

# if _str == "iter" or _str == "all":
#     gridsize = (31, 31)
#     g = Space(
#         [
#             numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
#             numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
#         ]
#     )
#     for _i in g:
#         print(_i)

#     h = Space(
#         [
#             -numpy.ones((3, 4), dtype=numpy.float32),
#             numpy.ones((3, 4), dtype=numpy.float32),
#         ]
#     )

#     for _h in h:
#         print(_h)
#         for __h in _h:
#             print(__h)
