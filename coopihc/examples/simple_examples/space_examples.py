import sys
from pathlib import Path

file = Path(__file__).resolve()
root = file.parents[3]
sys.path.append(str(root))

if __name__ == "__main__":

    import numpy
    from coopihc.helpers import flatten
    from coopihc.space.Space import Space

    numpy.set_printoptions(precision=3, suppress=True)

    # [start-space-def]
    continous_space = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ]
    )

    discrete_spaces = Space([numpy.array([1, 2, 3], dtype=numpy.int16)])
    # [end-space-def]

    # [start-space-complex-def]
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

    none_space = Space([numpy.array([None], dtype=object)])
    # [end-space-complex-def]

    # [start-space-contains]
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
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), numpy.float32),
        ]
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
    # [end-space-contains]

    # [start-space-sample]
    f = Space(
        [
            numpy.array([[-2, -2], [-1, -1]], dtype=numpy.float32),
            numpy.ones((2, 2), numpy.float32),
        ]
    )
    g = Space(
        [
            numpy.array([i for i in range(31)], dtype=numpy.int16),
            numpy.array([i for i in range(31)], dtype=numpy.int16),
        ]
    )
    h = Space([numpy.array([i for i in range(10)], dtype=numpy.int16)])

    f.sample()
    g.sample()
    h.sample()
    # [end-space-sample]

    # [start-space-iter]
    g = Space(
        [
            numpy.array([i for i in range(31)], dtype=numpy.int16),
            numpy.array([i for i in range(31)], dtype=numpy.int16),
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
    # [end-space-iter]
