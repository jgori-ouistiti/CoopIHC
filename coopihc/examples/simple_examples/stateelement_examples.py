import numpy
from coopihc.space.Space import Space
from coopihc.space.StateElement import StateElement
from coopihc.space.utils import (
    continuous_space,
    discrete_space,
    multidiscrete_space,
    autospace,
    StateNotContainedError,
    NumpyFunctionNotHandledWarning,
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

# [start-stateelement-array-function-not-defined]
cont_space = autospace([[-1, -1], [-1, -2]], [[1, 1], [1, 3]], dtype=numpy.float64)
g = StateElement(
    numpy.array([[0, 0.1], [-0.5, 0.8]], dtype=numpy.float64),
    cont_space,
    out_of_bounds_mode="warning",
)

# We would like to use numpy.amax(x)
# The function is not handled, so a warning is issued
with pytest.warns(NumpyFunctionNotHandledWarning):
    h = numpy.amax(g)

assert isinstance(h, numpy.ndarray)
assert not isinstance(h, StateElement)
assert h == 0.8
assert not hasattr(h, "spaces")
assert not hasattr(h, "out_of_bounds_mode")
# [end-stateelement-array-function-not-defined]

# [start-stateelement-array-function-define]
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
    obj = numpy.asarray(obj).reshape(1, 1).view(StateElement)
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


h = numpy.amax(g)
assert isinstance(y, StateElement)
assert StateElement.HANDLED_FUNCTIONS.get(numpy.amax) is not None
assert g.HANDLED_FUNCTIONS.get(numpy.amax) is not None
assert h.shape == (1, 1)
assert h == 0.8
assert h.spaces.space_type == "continuous"
assert h.spaces.shape == (1, 1)
assert h.spaces.low == numpy.array([[-2]])
assert h.spaces.high == numpy.array([[3]])
# [end-stateelement-array-function-define]


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
for i, _x in enumerate(x):
    for j, __x in enumerate(_x):
        assert isinstance(__x, StateElement)
        if i == 1 and j == 1:
            assert __x.equals(
                StateElement(numpy.array([[0]]), autospace([[-2]], [[3]])), mode="hard"
            )
        else:
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


# [start-stateelement-equal]
discr_space = discrete_space([1, 2, 3])
other_discr_space = discrete_space([1, 2, 3, 4])
x = StateElement(numpy.array(1), discr_space)
w = StateElement(numpy.array(1), other_discr_space)
assert w.equals(x)
assert not w.equals(x, "hard")
# [end-stateelement-equal]


# [start-stateelement-getitem]
x = StateElement(numpy.array([[0.0, 0.1], [0.2, 0.3]]), cont_space)
assert x[0, 0] == 0.0
assert x[0, 0, {"spaces": True}] == StateElement(
    numpy.array([[0.0]]), autospace(numpy.array([[-1]]), numpy.array([[1]]))
)
# [end-stateelement-getitem]


# [start-stateelement-cast]

# --------------------------- Init viz.

# import matplotlib.pyplot as plt

# fig = plt.figure()
# axd2c = fig.add_subplot(221)
# axc2d = fig.add_subplot(222)
# axc2c = fig.add_subplot(223)
# axd2d = fig.add_subplot(224)

# --------------------------  Discrete 2 Continuous
discr_space = autospace([1, 2, 3])
cont_space = autospace([[-1.5]], [[1.5]])

_center = []
_edges = []
for i in [1, 2, 3]:
    x = StateElement(i, discr_space)
    _center.append(x.cast(cont_space, mode="center").squeeze().tolist())
    _edges.append(x.cast(cont_space, mode="edges").squeeze().tolist())


# axd2c.plot(numpy.array([1, 2, 3]), numpy.array(_center) - 0.05, "+", label="center")
# axd2c.plot(numpy.array([1, 2, 3]), numpy.array(_edges) + 0.05, "o", label="edges")
# axd2c.legend()

# ------------------------   Continuous 2 Discrete
center = []
edges = []
for i in numpy.linspace(-1.5, 1.5, 100):
    x = StateElement(numpy.array(i).reshape((1, 1)), cont_space)

    ret_stateElem = x.cast(discr_space, mode="center")
    center.append(ret_stateElem[:].squeeze().tolist())

    ret_stateElem = x.cast(discr_space, mode="edges")
    edges.append(ret_stateElem[:].squeeze().tolist())


# axc2d.plot(
#     numpy.linspace(-1.5, 1.5, 100), numpy.array(center) - 0.05, "+", label="center"
# )
# axc2d.plot(
#     numpy.linspace(-1.5, 1.5, 100), numpy.array(edges) + 0.05, "o", label="edges"
# )
# axc2d.legend()


# ------------------------- Continuous2Continuous

cont_space = autospace(numpy.full((2, 2), -1), numpy.full((2, 2), 1))
other_cont_space = autospace(numpy.full((2, 2), 0), numpy.full((2, 2), 4))

output = []
for i in numpy.linspace(-1, 1, 100):
    x = StateElement(numpy.full((2, 2), i), cont_space)
    output.append(x.cast(other_cont_space)[0, 0].squeeze().tolist())

# axc2c.plot(numpy.linspace(-1, 1, 100), numpy.array(output), "-")

# -------------------------- Discrete2Discrete

discr_space = autospace([1, 2, 3, 4])
other_discr_space = autospace([11, 12, 13, 14])


output = []
for i in [1, 2, 3, 4]:
    x = StateElement(i, discr_space)
    output.append(x.cast(other_discr_space).squeeze().tolist())

# axd2d.plot([1, 2, 3, 4], output, "+")


# ----------------------- show viz
# plt.tight_layout()
# plt.show()

# [end-stateelement-cast]
