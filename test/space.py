from coopihc.space import Space
from coopihc.helpers import flatten
import numpy

import sys

_str = sys.argv[1]

if _str == "init" or _str == "all":
    s = Space([numpy.array([1, 2, 3], dtype=numpy.int16)])
    d = Space(
        [-numpy.ones((2, 2), dtype=numpy.float32), numpy.ones((2, 2), numpy.float32)]
    )
    f = Space(
        [
            numpy.array([[-2, -2], [-1, -1]], dtype=numpy.float32),
            numpy.ones((2, 2), numpy.float32),
        ]
    )

    gridsize = (31, 31)
    g = Space(
        [
            numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
            numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
        ]
    )
    number_of_targets = 1
    h = Space(
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

if _str == "action-space":
    h = Space(
        flatten(
            [
                [
                    numpy.array([i for i in range(4)], dtype=numpy.int16),
                    numpy.array([i for i in range(4)], dtype=numpy.int16),
                ]
                for j in range(3)
            ]
        )
    )
    s = Space([numpy.array([1, 2, 3], dtype=numpy.int16)])
    a = Space([numpy.array([None], dtype=object)])

if _str == "contains" or _str == "all":
    space = Space([numpy.array([1, 2, 3], dtype=numpy.int16)])
    x = numpy.array([2], dtype=numpy.int16)
    y = numpy.array([2], dtype=numpy.float32)
    yy = numpy.array([2])
    z = numpy.array([5])
    assert x in space
    assert y not in space
    assert yy in space
    assert z not in space

    space = Space(
        [-numpy.ones((2, 2), dtype=numpy.float32), numpy.ones((2, 2), numpy.float32)]
    )
    x = numpy.array([[1, 1], [1, 1]], dtype=numpy.int16)
    y = numpy.array([[1, 1], [1, 1]], dtype=numpy.float32)
    yy = numpy.array([[1, 1], [1, 1]])
    yyy = numpy.array([[1.0, 1.0], [1.0, 1.0]])
    z = numpy.array([[5, 1], [1, 1]], dtype=numpy.float32)
    assert x in space
    assert y in space
    assert yy in space
    assert yyy in space
    assert z not in space

    gridsize = (31, 31)
    g = Space(
        [
            numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
            numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
        ]
    )

    x = numpy.array([2, 3])
    xx = numpy.array([2, 3, 1])
    xxx = numpy.array([2.0, 3.0])
    xxxx = numpy.array([150, 21])

    assert x in g
    assert xx not in g
    assert xxx not in g
    assert xxxx not in g

if _str == "sample" or _str == "all":
    gridsize = (31, 31)
    f = Space(
        [
            numpy.array([[-2, -2], [-1, -1]], dtype=numpy.float32),
            numpy.ones((2, 2), numpy.float32),
        ]
    )
    g = Space(
        [
            numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
            numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
        ]
    )
    h = Space([numpy.array([i for i in range(10)], dtype=numpy.int16)])

    f.sample()
    g.sample()
    h.sample()

if _str == "iter" or _str == "all":
    gridsize = (31, 31)
    g = Space(
        [
            numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
            numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
        ]
    )
    for _i in g:
        print(_i)

    h = Space(
        [
            -numpy.ones((3, 4), dtype=numpy.float32),
            numpy.ones((3, 4), dtype=numpy.float32),
        ]
    )

    for _h in h:
        print(_h)
        for __h in _h:
            print(__h)
