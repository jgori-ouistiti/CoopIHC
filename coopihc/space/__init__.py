from .Space import Space
from .StateElement import StateElement
from .State import State
from .utils import remove_prefix, SpaceLengthError, StateNotContainedError

import numpy
from coopihc.helpers import flatten

numpy.set_printoptions(precision=3, suppress=True)


# ================ Some Examples ==============
if __name__ == "__main__":

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

    none_space = Space([numpy.array([None], dtype=numpy.object)])
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

    # [start-state-example]
    # Continuous substate. Provide Space([low, high]). Value is optional
    x = StateElement(
        values=None,
        spaces=Space(
            [
                numpy.array([-1.0]).reshape(1, 1),
                numpy.array([1.0]).reshape(1, 1),
            ]
        ),
    )

    # Discrete substate. Provide Space([range]). Value is optional
    y = StateElement(values=2, spaces=Space([numpy.array([1, 2, 3], dtype=numpy.int)]))

    # Define a State, composed of two substates previously defined
    s1 = State(substate_x=x, substate_y=y)

    # Define a super-State that is composed of the State previously defined
    S = State()
    S["substate1"] = s1
    # [end-state-example]

    # -------------- StateElement------------

    # [start-stateelement-init]
    x = StateElement(
        values=None,
        spaces=[
            Space(
                [
                    numpy.array([-1], dtype=numpy.float32),
                    numpy.array([1], dtype=numpy.float32),
                ]
            ),
            Space([numpy.array([1, 2, 3], dtype=numpy.int16)]),
            Space([numpy.array([-6, -5, -4, -3, -2, -1], dtype=numpy.int16)]),
        ],
    )

    y = StateElement(
        values=None,
        spaces=[
            Space(
                [
                    numpy.array([i for i in range(10)], dtype=numpy.int16),
                    numpy.array([i for i in range(10)], dtype=numpy.int16),
                ]
            )
            for j in range(3)
        ],
        clipping_mode="error",
    )
    # [end-stateelement-init]

    # [start-stateelement-reset]
    x.reset()
    y.reset()
    reset_dic = {"values": [-1 / 2, 2, -2]}
    x.reset(dic=reset_dic)
    reset_dic = {"values": [[0, 0], [10, 10], [5, 5]]}
    try:
        y.reset(dic=reset_dic)
    except StateNotContainedError:
        print("raised error as expected")
    # [end-stateelement-reset]

    # [start-stateelement-iter]
    for _x in x:
        print(_x)

    for _y in y:
        print(_y)
    # [end-stateelement-iter]

    # [start-stateelement-cp]
    x.reset()
    for n, _x in enumerate(x.cartesian_product()):
        # print(n, _x.values)
        print(n, _x)
    y.reset()
    # There are a million possible elements in y, so consider the first subspace only
    for n, _y in enumerate(y[0].cartesian_product()):
        print(n, _y.values)
    # [end-stateelement-cp]

    # [start-stateelement-comp]
    x.reset()
    a = x[0]
    print(x < numpy.array([2, -2, 4]))
    # [end-stateelement-comp]

    y.reset()
    targetdomain = StateElement(
        values=None,
        spaces=[
            Space(
                [
                    -numpy.ones((2, 1), dtype=numpy.float32),
                    numpy.ones((2, 1), dtype=numpy.float32),
                ]
            )
            for j in range(3)
        ],
    )
    res = y.cast(targetdomain)

    # [start-stateelement-cast]
    b = StateElement(
        values=5,
        spaces=Space(
            [numpy.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=numpy.int16)]
        ),
    )

    a = StateElement(
        values=0,
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    import matplotlib.pyplot as plt

    # C2D
    continuous = []
    discrete = []
    for elem in numpy.linspace(-1, 1, 200):
        a["values"] = elem
        continuous.append(a["values"][0].squeeze().tolist())
        discrete.append(a.cast(b, mode="center")["values"][0].squeeze().tolist())

    plt.plot(continuous, discrete, "b*")
    plt.show()

    # D2C

    continuous = []
    discrete = []
    for elem in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
        b["values"] = elem
        discrete.append(elem)
        continuous.append(b.cast(a, mode="edges")["values"][0].squeeze().tolist())

    plt.plot(discrete, continuous, "b*")
    plt.show()

    # C2C

    a = StateElement(
        values=0,
        spaces=Space(
            [
                numpy.array([-2], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    b = StateElement(
        values=3.5,
        spaces=Space(
            [
                numpy.array([3], dtype=numpy.float32),
                numpy.array([4], dtype=numpy.float32),
            ],
        ),
    )

    c1 = []
    c2 = []
    for elem in numpy.linspace(-2, 1, 100):
        a["values"] = elem
        c1.append(a["values"][0].squeeze().tolist())
        c2.append(a.cast(b)["values"][0].squeeze().tolist())

    plt.plot(c1, c2, "b*")
    plt.show()

    # D2D
    a = StateElement(
        values=5,
        spaces=Space([numpy.array([i for i in range(11)], dtype=numpy.int16)]),
    )
    b = StateElement(
        values=5,
        spaces=Space(
            [numpy.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=numpy.int16)]
        ),
    )

    d1 = []
    d2 = []
    for i in range(11):
        a["values"] = i
        d1.append(i)
        d2.append(a.cast(b)["values"][0].squeeze().tolist())

    plt.plot(d1, d2, "b*")
    plt.show()
    # [end-stateelement-cast]

    # [start-stateelement-arithmetic]
    # Neg
    x = StateElement(
        values=numpy.array([[-0.237]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    print(-x)

    # Sum
    x = StateElement(
        values=numpy.array([[-0.237]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    y = StateElement(
        values=numpy.array([[-0.135]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1], dtype=numpy.float32),
                numpy.array([1], dtype=numpy.float32),
            ]
        ),
    )
    # adding two simple StateElements
    print(x + y)
    # add with a scalar
    z = -0.5
    print(x + z)

    a = StateElement(
        values=numpy.array([[-0.237, 0]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1, -1], dtype=numpy.float32),
                numpy.array([1, 1], dtype=numpy.float32),
            ]
        ),
    )
    b = StateElement(
        values=numpy.array([[0.5, 0.5]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1, -1], dtype=numpy.float32),
                numpy.array([1, 1], dtype=numpy.float32),
            ]
        ),
    )

    print(a + b)
    print(b + a)
    print(a - b)
    print(b - a)
    c = numpy.array([0.12823329, -0.10512559])
    print(a + c)
    print(c + a)
    print(a - c)
    print(c - a)

    # Mul
    a = StateElement(
        values=numpy.array([[-0.237, 0]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1, -1], dtype=numpy.float32),
                numpy.array([1, 1], dtype=numpy.float32),
            ]
        ),
    )
    b = StateElement(
        values=numpy.array([[0.5, 0.5]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1, -1], dtype=numpy.float32),
                numpy.array([1, 1], dtype=numpy.float32),
            ]
        ),
    )

    c = 1
    print(a * b)
    print(a * c)
    print(b * a)
    print(c * a)

    # Matmul
    a = StateElement(
        values=numpy.array([[-0.237, 0], [1, 1]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([[-1, -1], [-1, -1]], dtype=numpy.float32),
                numpy.array([[1, 1], [1, 1]], dtype=numpy.float32),
            ]
        ),
    )
    b = StateElement(
        values=numpy.array([[0.5, 0.5]], dtype=numpy.float32),
        spaces=Space(
            [
                numpy.array([-1, -1], dtype=numpy.float32).reshape(-1, 1),
                numpy.array([1, 1], dtype=numpy.float32).reshape(-1, 1),
            ]
        ),
    )

    z = numpy.ones((2, 2))

    print(a @ b)
    print(z @ a)
    print(a @ z)
    # [end-stateelement-arithmetic]

    # -------------- State------------

    # [start-state-init]
    x = StateElement(
        values=1,
        spaces=Space(
            [
                numpy.array([-1.0]).reshape(1, 1),
                numpy.array([1.0]).reshape(1, 1),
            ]
        ),
    )

    y = StateElement(values=2, spaces=Space([numpy.array([1, 2, 3], dtype=numpy.int)]))

    z = StateElement(
        values=5,
        spaces=Space([numpy.array([i for i in range(10)], dtype=numpy.int)]),
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
        values=None,
        spaces=Space(
            [numpy.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6], dtype=numpy.int16)]
        ),
    )

    s2 = State(**{"substate_xx": xx, "substate_yy": yy})

    S = State()
    S["substate1"] = s1
    S["substate2"] = s2
    # [end-state-init]

    # [start-state-reset]
    print(S.reset())
    # [end-state-reset]

    # [start-state-filter]
    from collections import OrderedDict

    ordereddict = OrderedDict(
        {"substate1": OrderedDict({"substate_x": 0, "substate_w": 0})}
    )

    ns1 = S.filter("values", filterdict=ordereddict)
    ns2 = S.filter("spaces", filterdict=ordereddict)
    ns5 = S.filter("values")
    ns6 = S.filter("spaces")

    # [end-state-filter]

    # [start-state-serialize]
    print(S.serialize())
    # [end-state-serialize]
