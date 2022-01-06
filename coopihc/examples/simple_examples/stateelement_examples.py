import numpy
from coopihc.space.Space import Space
from coopihc.space.StateElement import StateElement
from coopihc.space.utils import (
    continuous_space,
    discrete_space,
    multidiscrete_space,
    autospace,
    StateNotContainedError,
)
import pytest

numpy.set_printoptions(precision=3, suppress=True)

# [start-stateelement-init]
discr_space = discrete_space([1, 2, 3])
x = StateElement([2], discr_space, out_of_bounds_mode="error")

cont_space = continuous_space(-numpy.ones((2, 2)), numpy.ones((2, 2)))
y = StateElement(numpy.zeros((2, 2)), cont_space, out_of_bounds_mode="error")

multidiscr_space = multidiscrete_space(
    [
        numpy.array([1, 2, 3]),
        numpy.array([1, 2, 3, 4, 5]),
        numpy.array([1, 3, 5, 8]),
    ]
)
z = StateElement([1, 5, 3], multidiscr_space, out_of_bounds_mode="error")
# [end-stateelement-init]

# [start-stateelement-ufunc]
discr_space = discrete_space([1, 2, 3])
x = StateElement(2, discr_space, out_of_bounds_mode="error")
assert x + numpy.array(1) == 3
assert x + 1 == 3
try:
    x + 2
except StateNotContainedError:
    print("Exception as expected")
# [end-stateelement-ufunc]

# [start-stateelement-array-function]
cont_space = autospace([[-1, -1], [-1, -2]], [[1, 1], [1, 3]], dtype=numpy.float64)
x = StateElement(
    numpy.array([[0, 0.1], [-0.5, 0.8]], dtype=numpy.float64),
    cont_space,
    out_of_bounds_mode="warning",
)

# We would like to use numpy.amax(x)
# The function is not handled, so a warning is issued
with pytest.warns(NumpyFunctionNotHandledWarning):
    y = numpy.amax(x)

assert isinstance(y, numpy.ndarray)
assert not isinstance(y, StateElement)
assert y == 0.8
assert not hasattr(y, "spaces")
assert not hasattr(y, "out_of_bounds_mode")


@StateElement.implements(numpy.amax)
def amax(arr, **keywordargs):
    spaces, out_of_bounds_mode, kwargs = (
        arr.spaces,
        arr.out_of_bounds_mode,
        arr.kwargs,
    )
    obj = arr.view(numpy.ndarray)
    argmax = numpy.argmax(obj, **keywordargs)
    index = numpy.unravel_index(argmax, arr.spaces.shape)
    obj = numpy.amax(obj, **keywordargs)
    obj = numpy.asarray(obj).view(StateElement)
    if arr.spaces.space_type == "continuous":
        obj.spaces = autospace(
            numpy.atleast_2d(arr.spaces.low[index[0], index[1]]),
            numpy.atleast_2d(arr.spaces.high[index[0], index[1]]),
        )
    else:
        raise NotImplementedError
    obj.out_of_bounds_mode = arr.out_of_bounds_mode
    obj.kwargs = arr.kwargs
    return obj


y = numpy.amax(x)
assert isinstance(y, StateElement)
assert StateElement.HANDLED_FUNCTIONS.get(numpy.amax) is not None
assert x.HANDLED_FUNCTIONS.get(numpy.amax) is not None
assert y.shape == ()
assert y == 0.8
assert y.spaces.space_type == "continuous"
assert y.spaces.shape == (1, 1)
assert y.spaces.low == numpy.array([[-2]])
assert y.spaces.high == numpy.array([[3]])
# [end-stateelement-array-function]


# [start-stateelement-reset]

# Random resets
x.reset()
y.reset()
z.reset()

# Forced resets
x.reset(value=2)
# you don't actually have to specify the value keyword
y.reset(0.59 * numpy.ones((2, 2)))
z.reset([1, 1, 8])

# forced reset also raise input checks:
try:
    x.reset(value=5)
except StateNotContainedError:
    print("raised Error {} as expected".format(StateNotContainedError))

# [end-stateelement-reset]

# [start-stateelement-iter]
# iterating over a discrete space returns itself
x = StateElement([2], discr_space, out_of_bounds_mode="error")
for _x in x:
    assert _x == x

# iterating over a continuous space: first over rows, then over columns
x = StateElement(numpy.zeros((2, 2)), cont_space)
for _x in x:
    for __x in _x:
        assert isinstance(__x, StateElement)
        assert __x.equals(
            StateElement(numpy.array([[0]]), autospace([[-1]], [[1]])), mode="hard"
        )

# iterating over a multidiscrete space returns discrete spaces
multidiscr_space = multidiscrete_space(
    [
        numpy.array([1, 2]),
        numpy.array([1, 2, 3, 4, 5]),
        numpy.array([1, 3, 5, 8]),
    ]
)
x = StateElement(numpy.array([[1], [1], [3]]), multidiscr_space)
for n, _x in enumerate(x):
    assert _x.spaces.space_type == "discrete"
# [end-stateelement-iter]


# [start-stateelement-comp]
x.reset()
x < 4
# [end-stateelement-comp]

# y.reset()
# targetdomain = StateElement(
#     values=None,
#     spaces=[
#         Space(
#             [
#                 -numpy.ones((2, 1), dtype=numpy.float32),
#                 numpy.ones((2, 1), dtype=numpy.float32),
#             ]
#         )
#         for j in range(3)
#     ],
# )
# res = y.cast(targetdomain)

# b = StateElement(
#     values=5,
#     spaces=Space(
#         [numpy.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=numpy.nt16)]
#     ),
# )

# a = StateElement(
#     values=0,
#     spaces=Space(
#         [
#             numpy.array([-1], dtype=numpy.float32),
#             numpy.array([1], dtype=numpy.float32),
#         ]
#     ),
# )
# import matplotlib.pyplot as plt

# # C2D
# continuous = []
# discrete = []
# for elem in numpy.linspace(-1, 1, 200):
#     a["values"] = elem
#     continuous.append(a["values"][0].squeeze().tolist())
#     discrete.append(a.cast(b, mode="center")["values"][0].squeeze().tolist())

# plt.plot(continuous, discrete, "b*")
# plt.show()

# # D2C

# continuous = []
# discrete = []
# for elem in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
#     b["values"] = elem
#     discrete.append(elem)
#     continuous.append(b.cast(a, mode="edges")["values"][0].squeeze().tolist())

# plt.plot(discrete, continuous, "b*")
# plt.show()

# # C2C

# a = StateElement(
#     values=0,
#     spaces=Space(
#         [
#             numpy.array([-2], dtype=numpy.float32),
#             numpy.array([1], dtype=numpy.float32),
#         ]
#     ),
# )
# b = StateElement(
#     values=3.5,
#     spaces=Space(
#         [
#             numpy.array([3], dtype=numpy.float32),
#             numpy.array([4], dtype=numpy.float32),
#         ],
#     ),
# )

# c1 = []
# c2 = []
# for elem in numpy.linspace(-2, 1, 100):
#     a["values"] = elem
#     c1.append(a["values"][0].squeeze().tolist())
#     c2.append(a.cast(b)["values"][0].squeeze().tolist())

# plt.plot(c1, c2, "b*")
# plt.show()

# # D2D
# a = StateElement(
#     values=5, spaces=Space([numpy.array([i for i in range(11)], dtype=numpy.int16)])
# )
# b = StateElement(
#     values=5,
#     spaces=Space(
#         [numpy.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=numpy.nt16)]
#     ),
# )

# d1 = []
# d2 = []
# for i in range(11):
#     a["values"] = i
#     d1.append(i)
#     d2.append(a.cast(b)["values"][0].squeeze().tolist())

# plt.plot(d1, d2, "b*")
# plt.show()


# [start-stateelement-cast]

# TODO

# [end-stateelement-cast]

# [start-stateelement-arithmetic]

# TODO

# [end-stateelement-arithmetic]
