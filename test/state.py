import sys

_str = sys.argv[1]

import coopihc
from coopihc.space import StateElement, Space, State
import numpy


x = StateElement(
    values=1,
    spaces=Space([numpy.array([-1.0]).reshape(1, 1), numpy.array([1.0]).reshape(1, 1)]),
)

y = StateElement(values=2, spaces=Space(numpy.array([1, 2, 3], dtype=numpy.int)))

z = StateElement(
    values=5, spaces=Space(numpy.array([i for i in range(10)], dtype=numpy.int))
)


s1 = State(substate_x=x, substate_y=y, substate_z=z)

w = StateElement(
    values=numpy.zeros((3, 3)),
    spaces=Space([-3.5 * numpy.ones((3, 3)), 6 * numpy.ones((3, 3))]),
)
s1["substate_w"] = w

xx = StateElement(
    values=numpy.ones((2, 2)),
    spaces=Space([-0.5 * numpy.ones((2, 2)), 0.5 * numpy.ones((2, 2))]),
    clipping_mode="clip",
)

yy = StateElement(
    values=None, spaces=Space(numpy.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]))
)


s2 = State(**{"substate_xx": xx, "substate_yy": yy})

S = State()
S["substate1"] = s1
S["substate2"] = s2


if _str == "reset" or _str == "all":
    print(S.reset())

if _str == "flat" or _str == "all":
    print(S.flat())

if _str == "repr" or _str == "all":
    print(S)

if _str == "filter" or _str == "all":

    from collections import OrderedDict

    ordereddict = OrderedDict(
        {"substate1": OrderedDict({"substate_x": 0, "substate_w": 0})}
    )

    ns1 = S.filter("values", filterdict=ordereddict)
    ns2 = S.filter("spaces", filterdict=ordereddict)
    ns5 = S.filter("values")
    ns6 = S.filter("spaces")


if _str == "copy" or _str == "all":

    import copy
    import time

    start = time.time()
    for i in range(1000):
        _copy = copy.copy(S)
    mid = time.time()
    for i in range(1000):
        _deepcopy = copy.deepcopy(S)
    end = time.time()
    print(mid - start)
    print(end - start)


if _str == "serialize" or _str == "all":

    print(S.serialize())
