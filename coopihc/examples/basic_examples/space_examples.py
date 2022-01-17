import numpy
from coopihc.helpers import flatten
from coopihc.space.Space import Space
from coopihc.space.utils import (
    discrete_space,
    continuous_space,
    multidiscrete_space,
    autospace,
)


numpy.set_printoptions(precision=3, suppress=True)

# [start-space-def]
# Continuous space
cont_space = Space(
    [
        -numpy.ones((2, 2), dtype=numpy.float32),
        numpy.ones((2, 2), dtype=numpy.float32),
    ],
    "continuous",
)
# Shortcut
cont_space = continuous_space(-numpy.ones((2, 2)), numpy.ones((2, 2)))


# Discrete space
discr_space = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
# Shortcut
discr_space = discrete_space(numpy.array([1, 2, 3]))

# Multidiscrete space
mult_space = Space(
    [
        numpy.array([1, 2, 3], dtype=numpy.int16),
        numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
    ],
    "multidiscrete",
)
mult_space = multidiscrete_space([numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])])


# [end-space-def]

# [start-space-complex-def]
multidiscrete_space = Space(
    flatten(
        [
            [
                numpy.array([i for i in range(4)], dtype=numpy.int16),
                numpy.array([i for i in range(5)], dtype=numpy.int16),
            ]
            for j in range(3)
        ]
    ),
    "multidiscrete",
)
# [end-space-complex-def]

# [start-space-autospace]
# Space(numpy.array([1, 2, 3]), "discrete")
autospace([1, 2, 3])
autospace([[1, 2, 3]])
autospace(numpy.array([1, 2, 3]))
autospace(numpy.array([[1, 2, 3]]))
autospace([numpy.array([1, 2, 3])])
autospace([numpy.array([[1, 2, 3]])])

# Space([numpy.array([1, 2, 3]), numpy.array([4, 5, 6])], "multidiscrete")
autospace(numpy.array([[1, 2, 3], [4, 5, 6]]))
autospace([1, 2, 3], [1, 2, 3, 4, 5])
autospace([numpy.array([1, 2, 3])], [numpy.array([1, 2, 3, 4, 5])])
autospace(numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5]))
autospace([numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])])
autospace([numpy.array([[1, 2, 3]]), numpy.array([[1, 2, 3, 4, 5]])])
autospace([1, 2, 3], [1, 2, 3, 4, 5], [1, 8])
autospace(
    [
        numpy.array([[1, 2, 3]]),
        numpy.array([[1, 2, 3, 4, 5]]),
        numpy.array([[1, 8]]),
    ]
)

# Space([-numpy.ones((2, 2)), numpy.ones((2, 2))], "continuous")
autospace(-numpy.array([[1, 1], [1, 1]]), numpy.array([[1, 1], [1, 1]]))
autospace([-numpy.array([[1, 1], [1, 1]]), numpy.array([[1, 1], [1, 1]])])
autospace([[-1, -1], [-1, -1]], [[1, 1], [1, 1]])
autospace([[[-1, -1], [-1, -1]], [[1, 1], [1, 1]]])


# [end-space-autospace]


# [start-space-contains]
space = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
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
        numpy.ones((2, 2), dtype=numpy.float32),
    ],
    "continuous",
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
    ],
    "continuous",
)
g = Space(
    [
        numpy.array([i for i in range(31)], dtype=numpy.int16),
        numpy.array([i for i in range(31)], dtype=numpy.int16),
    ],
    "multidiscrete",
)
h = Space(numpy.array([i for i in range(10)], dtype=numpy.int16), "discrete")

f.sample()
# >>> Space([[[-2. -2.]
#  [-1. -1.]], [[1. 1.]
#  [1. 1.]]], 'continuous', contains = 'soft')
g.sample()
# >>> Space([[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30],[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
#  24 25 26 27 28 29 30]], 'multidiscrete', contains = 'soft')
h.sample()
# >>> Space([0 1 2 3 4 5 6 7 8 9], 'discrete', contains = 'soft')
# [end-space-sample]

# [start-space-iter]
g = Space(
    [
        numpy.array([i for i in range(31)], dtype=numpy.int16),
        numpy.array([i for i in range(31)], dtype=numpy.int16),
    ],
    "multidiscrete",
)
for _i in g:
    # print(_i)
    pass

h = Space(
    [
        -numpy.ones((3, 4), dtype=numpy.float32),
        numpy.ones((3, 4), dtype=numpy.float32),
    ],
    "continuous",
)
for _h in h:
    # print(_h)
    for __h in _h:
        # print(__h)
        pass
# [end-space-iter]

# [start-space-cp]
s = Space(
    numpy.array([i for i in range(3)], dtype=numpy.int16),
    "discrete",
)
q = Space(
    [
        numpy.array([i + 6 for i in range(2)], dtype=numpy.int16),
        numpy.array([i + 6 for i in range(2)], dtype=numpy.int16),
    ],
    "multidiscrete",
)
r = Space(
    [
        -numpy.ones((2, 2), dtype=numpy.float32),
        numpy.ones((2, 2), dtype=numpy.float32),
    ],
    "continuous",
)
cp, shape = Space.cartesian_product(s, q, r)
# cp
# >>> [[0 6 6 None]
#     [0 6 7 None]
#     [0 7 6 None]
#     [0 7 7 None]
#     [1 6 6 None]
#     [1 6 7 None]
#     [1 7 6 None]
#     [1 7 7 None]
#     [2 6 6 None]
#     [2 6 7 None]
#     [2 7 6 None]
#     [2 7 7 None]]

# shape
# >>> [(1,), (2, 1), (2, 2)]


# [end-space-cp]
