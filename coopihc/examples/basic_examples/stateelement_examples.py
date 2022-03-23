import numpy
from coopihc.base.elements import discrete_array_element, array_element, cat_element
from coopihc.base.utils import (
    StateNotContainedError,
)
import pytest

numpy.set_printoptions(precision=3, suppress=True)

# [start-stateelement-init]
# Value in a Categorical set
x = cat_element(3)
# Discrete value
y = discrete_array_element(low=2, high=5, init=2)
# Continuous Value
z = array_element(
    low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)), out_of_bounds_mode="error"
)
# [end-stateelement-init]

# [start-stateelement-ufunc]
x = cat_element(4, init=3, out_of_bounds_mode="error")
assert x + numpy.array(1) == 4
assert x + 1 == 4
try:
    x + 2
except StateNotContainedError:
    print("Exception as expected")
# [end-stateelement-ufunc]

# # [start-stateelement-array-function-not-defined]
# cont_space = autospace([[-1, -1], [-1, -2]], [[1, 1], [1, 3]], dtype=numpy.float64)
# g = StateElement(
#     numpy.array([[0, 0.1], [-0.5, 0.8]], dtype=numpy.float64),
#     cont_space,
#     out_of_bounds_mode="warning",
# )

# # We would like to use numpy.amax(x)
# # The function is not handled, so a warning is issued
# with pytest.warns(NumpyFunctionNotHandledWarning):
#     h = numpy.amax(g)

# assert isinstance(h, numpy.ndarray)
# assert not isinstance(h, StateElement)
# assert h == 0.8
# assert not hasattr(h, "spaces")
# assert not hasattr(h, "out_of_bounds_mode")
# # [end-stateelement-array-function-not-defined]

# # [start-stateelement-array-function-define]
# @StateElement.implements(numpy.amax)
# def amax(arr, **keywordargs):
#     spaces, out_of_bounds_mode, kwargs = (
#         arr.spaces,
#         arr.out_of_bounds_mode,
#         arr.kwargs,
#     )
#     obj = arr.view(numpy.ndarray)
#     argmax = numpy.argmax(obj, **keywordargs)
#     index = numpy.unravel_index(argmax, arr.spaces.shape)
#     obj = numpy.amax(obj, **keywordargs)
#     obj = numpy.asarray(obj).reshape(1, 1).view(StateElement)
#     if arr.spaces.space_type == "continuous":
#         obj.spaces = autospace(
#             numpy.atleast_2d(arr.spaces.low[index[0], index[1]]),
#             numpy.atleast_2d(arr.spaces.high[index[0], index[1]]),
#         )
#     else:
#         raise NotImplementedError
#     obj.out_of_bounds_mode = arr.out_of_bounds_mode
#     obj.kwargs = arr.kwargs
#     return obj


# h = numpy.amax(g)
# assert isinstance(y, StateElement)
# assert StateElement.HANDLED_FUNCTIONS.get(numpy.amax) is not None
# assert g.HANDLED_FUNCTIONS.get(numpy.amax) is not None
# assert h.shape == (1, 1)
# assert h == 0.8
# assert h.spaces.space_type == "continuous"
# assert h.spaces.shape == (1, 1)
# assert h.spaces.low == numpy.array([[-2]])
# assert h.spaces.high == numpy.array([[3]])
# # [end-stateelement-array-function-define]


# [start-stateelement-reset]

# Random resets
x.reset()
y.reset()
z.reset()

# Forced resets
x.reset(value=2)
y.reset(2)
z.reset(0.59 * numpy.ones((2, 2)))


# forced reset also raise input checks:
try:
    x.reset(value=5)
except StateNotContainedError:
    print("raised Error {} as expected".format(StateNotContainedError))

# [end-stateelement-reset]

# [start-stateelement-iter]
# iterating over a continuous space like in Numpy: first over rows, then over columns
x = array_element(
    init=numpy.array([[0.2, 0.3], [0.4, 0.5]]),
    low=-numpy.ones((2, 2)),
    high=numpy.ones((2, 2)),
)

for i, _x in enumerate(x):
    for j, _xx in enumerate(_x):
        print(_xx)

# you can't iterate over a discrete set
x = cat_element(4)
with pytest.raises(TypeError):
    next(iter(x))
# [end-stateelement-iter]

# [start-stateelement-comp]
x.reset()
x < 4
# [end-stateelement-comp]


# [start-stateelement-equal]
x = discrete_array_element(init=1, low=1, high=4)
w = discrete_array_element(init=1, low=1, high=3)
assert w.equals(x)
# Hard comparison checks also space
assert not w.equals(x, "hard")
# [end-stateelement-equal]


# [start-stateelement-getitem]
# CoopIHC abuses Python's indexing mechanism; you can extract values together with spaces
x = cat_element(3)
assert x[..., {"space": True}] == x
assert x[...] == x

x = array_element(
    init=numpy.array([[0.0, 0.1], [0.2, 0.3]]),
    low=-numpy.ones((2, 2)),
    high=numpy.ones((2, 2)),
)
assert x[0, 0] == 0.0
assert x[0, 0, {"space": True}] == array_element(init=0.0, low=-1, high=1)
# [end-stateelement-getitem]


# [start-stateelement-cast]

# --------------------------- Init viz.

import matplotlib.pyplot as plt

fig = plt.figure()
axd2c = fig.add_subplot(221)
axc2d = fig.add_subplot(222)
axc2c = fig.add_subplot(223)
axd2d = fig.add_subplot(224)

# --------------------------  Discrete 2 Continuous

_center = []
_edges = []
for i in [1, 2, 3]:
    x = discrete_array_element(init=i, low=1, high=3)
    y = array_element(init=i, low=-1.5, high=1.5)
    _center.append(x.cast(y, mode="center").tolist())
    _edges.append(x.cast(y, mode="edges").tolist())


axd2c.plot(numpy.array([1, 2, 3]), numpy.array(_center) - 0.05, "+", label="center")
axd2c.plot(numpy.array([1, 2, 3]), numpy.array(_edges) + 0.05, "o", label="edges")
axd2c.legend()

# ------------------------   Continuous 2 Discrete
center = []
edges = []
for i in numpy.linspace(-1.5, 1.5, 100):
    x = discrete_array_element(init=i, low=1, high=3)
    y = array_element(init=i, low=-1.5, high=1.5)

    ret_stateElem = y.cast(x, mode="center")
    center.append(ret_stateElem.tolist())

    ret_stateElem = y.cast(x, mode="edges")
    edges.append(ret_stateElem.tolist())


axc2d.plot(
    numpy.linspace(-1.5, 1.5, 100), numpy.array(center) - 0.05, "+", label="center"
)
axc2d.plot(
    numpy.linspace(-1.5, 1.5, 100), numpy.array(edges) + 0.05, "o", label="edges"
)
axc2d.legend()


# ------------------------- Continuous2Continuous

output = []
for i in numpy.linspace(-1, 1, 100):
    x = array_element(
        init=numpy.full((2, 2), i),
        low=numpy.full((2, 2), -1),
        high=numpy.full((2, 2), 1),
        dtype=numpy.float32,
    )
    y = array_element(
        low=numpy.full((2, 2), 0), high=numpy.full((2, 2), 4), dtype=numpy.float32
    )
    output.append(x.cast(y)[0, 0].tolist())

axc2c.plot(numpy.linspace(-1, 1, 100), numpy.array(output), "-")

# -------------------------- Discrete2Discrete


output = []
for i in [1, 2, 3, 4]:
    x = discrete_array_element(init=i, low=1, high=4)
    y = discrete_array_element(init=11, low=11, high=14)
    output.append(x.cast(y).tolist())

axd2d.plot([1, 2, 3, 4], output, "+")


# ----------------------- show viz
plt.tight_layout()
# plt.show()

# [end-stateelement-cast]
