import numpy

# from coopihc.helpers import flatten
from coopihc.base.Space import Space

from coopihc.base.elements import box_space, integer_set, integer_space


numpy.set_printoptions(precision=3, suppress=True)

# [start-space-def]
# Continuous space
cont_space = Space(
    low=-numpy.ones((2, 2)),
    high=numpy.ones((2, 2)),
)
# Shortcut
cont_space = box_space(numpy.ones((2, 2)))


# Discrete set
discr_set = Space(array=numpy.array([0, 1, 2, 3]))
# Shortcut
discr_set = integer_set(4)

# # Other shortcuts
# currently unavailable
# space = lin_space(-5, 5, num=11, dtype=numpy.int16)
# space = lin_space(-5, 5, num=22)
space = integer_space(10, dtype=numpy.int16)
space = box_space(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)))
# [end-space-def]


# [start-space-contains]
s = Space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
assert 1 in s
assert [1] in s
assert [[1]] in s
assert numpy.array(1) in s
assert numpy.array([1]) in s
assert numpy.array([[1]]) in s
assert 4 not in s
assert -1 not in s

# [end-space-contains]

# [start-space-sample]
s = Space(array=numpy.arange(1000), seed=123)
q = Space(array=numpy.arange(1000), seed=123)
r = Space(array=numpy.arange(1000), seed=12)
_s, _q, _r = s.sample(), q.sample(), r.sample()
assert _s in s
assert _q in q
assert _r in r
assert _s == _q
assert _s != _r
# [end-space-sample]


# [start-space-cp]
s = Space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
q = Space(array=numpy.array([-3, -2, -1], dtype=numpy.int16))
cp, shape = Space.cartesian_product(s, q)
# [end-space-cp]
