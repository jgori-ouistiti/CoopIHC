import core
from core.space import StateElement, State, StateNotContainedError, Space
import gym
import numpy
import sys
import copy


_str = sys.argv[1]

# -------- Correct assigment
if _str == 'correct' or _str == 'all':

    x = StateElement(
        values = None,
        spaces = [
            core.space.Space([
            numpy.array([-1], dtype = numpy.float32), numpy.array([1], dtype = numpy.float32)
            ]),
            core.space.Space(   [numpy.array([1,2,3], dtype =    numpy.int16)]),
            core.space.Space(   [numpy.array([-6,-5,-4,-3,-2,-1], dtype =    numpy.int16)])
                ]
    )


    gridsize = (11,11)
    number_of_targets = 3
    y = StateElement(
                    values = None,
                    spaces = [Space(
                    [numpy.array([i for i in range(gridsize[0])], dtype = numpy.int16),
                    numpy.array([i for i in range(gridsize[1])], dtype = numpy.int16)]
                                    ) for j in range(number_of_targets)],
                    clipping_mode = 'error')




x = StateElement(
    values = None,
    spaces = [
        core.space.Space([
        numpy.array([-1], dtype = numpy.float32), numpy.array([1], dtype = numpy.float32)
        ]),
        core.space.Space(   [numpy.array([1,2,3], dtype =    numpy.int16)]  ),
        core.space.Space(   [numpy.array([-6,-5,-4,-3,-2,-1], dtype =    numpy.int16)]  )
            ]
)


gridsize = (11,11)
number_of_targets = 3
y = StateElement(
                values = None,
                spaces = [Space(
                [numpy.array([i for i in range(gridsize[0])], dtype = numpy.int16),
                numpy.array([i for i in range(gridsize[1])], dtype = numpy.int16)]
                                ) for j in range(number_of_targets)],
                clipping_mode = 'error')

# -------------- accessing values
if _str == 'access' or _str == 'all':

    x['values']
    x['spaces']
    x.values
    x['values'] =  [numpy.array([[0.335]]), numpy.array([[2]]), numpy.array([[-4]])]
    x['values'] =  [0.2, 2, -5]

    y['values']
    y['spaces']
    y['values'] =  [    numpy.array([1,1]),
                        numpy.array([0,0]),
                        numpy.array([2,2])
                        ]
    y['values'] =  [   numpy.array([15,15]),
                        numpy.array([0,0]),
                        numpy.array([2,2])
                            ]

# ------ normal reset
if _str == 'reset' or _str == 'all':
    x.reset()
    y.reset()

# -------- forced reset
if _str == 'forced-reset' or _str == 'all':

    reset_dic = {'values': [-1/2,2,-2]}
    x.reset(dic = reset_dic)
    reset_dic = {'values': [[0,0], [10,10], [5,5]]}
    y.reset(dic = reset_dic)

# ------ iterate on StateElement
if _str == 'iter' or _str == 'all':

    for _x in x:
        print(_x)

    for _y in y:
        print(_y)

if _str == 'cartesianproduct' or _str == 'all':
    x.reset()
    for n,_x in enumerate(x.cartesian_product()):
        # print(n, _x.values)
        print(n, _x)
    y.reset()
    # There are a million elements in y
    for n,_y in enumerate(y[0].cartesian_product()):
        print(n, _y.values)


if _str == 'comp' or _str == 'all':
    x.reset()
    a = x[0]
    print(x < numpy.array([2,-2,4]))

if _str == 'len' or _str == 'all':
    len(x)
    len(y)

if _str == 'cast' or _str == 'all':

    b = StateElement(
        values = 5,
        spaces = core.space.Space(
            [numpy.array([-5,-4,-3,-2,-1,0,1,2,3,4,5], dtype = numpy.int16)]
        )
    )

    a = StateElement(
        values = 0,
        spaces =
            core.space.Space([
            numpy.array([-1], dtype = numpy.float32), numpy.array([1], dtype = numpy.float32)
            ]),
    )


    y.reset()
    targetdomain = StateElement(
        values = None,
        spaces = [
            core.space.Space([
                -numpy.ones((2,1), dtype = numpy.float32),
                numpy.ones((2,1), dtype = numpy.float32)
             ]) for j in range(3)
        ]
    )
    res = y.cast(targetdomain)
    # C2D
    continuous = []
    discrete = []
    for elem in numpy.linspace(-1,1,200):
        a['values'] = elem
        continuous.append(a['values'][0].squeeze().tolist())
        discrete.append(a.cast(b, mode = 'center')['values'][0].squeeze().tolist())
    import matplotlib.pyplot as plt
    plt.plot(continuous, discrete, 'b*')
    plt.show()

    # D2C

    continuous = []
    discrete = []
    for elem in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
        b['values'] = elem
        discrete.append(elem)
        continuous.append(b.cast(a, mode = 'edges')['values'][0].squeeze().tolist())
    import matplotlib.pyplot as plt
    plt.plot(discrete, continuous, 'b*')
    plt.show()

    # C2C

    a = StateElement(
        values = 0,
        spaces =
            core.space.Space([
            numpy.array([-2], dtype = numpy.float32), numpy.array([1], dtype = numpy.float32)
            ]),
    )
    b = StateElement(
        values = 3.5,
        spaces =
            core.space.Space([
            numpy.array([3], dtype = numpy.float32), numpy.array([4], dtype = numpy.float32)
            ]),
    )
    c1 = []
    c2 = []
    for elem in numpy.linspace(-2,1, 100):
        a['values'] = elem
        c1.append(a['values'][0].squeeze().tolist())
        c2.append(a.cast(b)['values'][0].squeeze().tolist())
    import matplotlib.pyplot as plt
    plt.plot(c1, c2, 'b*')
    plt.show()

    # D2D
    a = StateElement(
        values = 5,
        spaces = core.space.Space(
            [numpy.array([i for i in range(11)], dtype = numpy.int16)]
        )
    )
    b = StateElement(
        values = 5,
        spaces = core.space.Space(
            [numpy.array([-5,-4,-3,-2,-1,0,1,2,3,4,5], dtype = numpy.int16)]
        )
    )

    d1 = []
    d2 = []
    for i in range(11):
        a['values'] = i
        d1.append(i)
        d2.append(a.cast(b)['values'][0].squeeze().tolist())
    import matplotlib.pyplot as plt
    plt.plot(d1, d2, 'b*')
    plt.show()

if _str == 'neg' or _str == 'all':
    x = StateElement(
        values = numpy.array([[-0.237]], dtype=numpy.float32),
        spaces =
            core.space.Space([
            numpy.array([-1], dtype = numpy.float32), numpy.array([1], dtype = numpy.float32)
            ]),
    )
    print(-x)


if _str == 'typing_priority' or _str == 'all':
    x = StateElement(
        values = numpy.array([[-0.237]], dtype=numpy.float16),
        spaces =
            core.space.Space([
            numpy.array([-1], dtype = numpy.float32), numpy.array([1], dtype = numpy.float32)
            ]),
    )

    y = StateElement(
        values = numpy.array([[-0.237]], dtype=numpy.float16),
        spaces =
            core.space.Space([
            numpy.array([-1], dtype = numpy.float32), numpy.array([1], dtype = numpy.float32)
            ]),
        typing_priority = 'value'
    )

if _str == 'add' or _str == 'all':
    x = StateElement(
        values = numpy.array([[-0.237]], dtype=numpy.float32),
        spaces =
            core.space.Space([
            numpy.array([-1], dtype = numpy.float32), numpy.array([1], dtype = numpy.float32)
            ]),
    )
    y = StateElement(
        values = numpy.array([[-0.135]], dtype=numpy.float32),
        spaces =
            core.space.Space([
            numpy.array([-1], dtype = numpy.float32), numpy.array([1], dtype = numpy.float32)
            ]),
    )
    # adding two simple StateElements
    print(x + y)
    # add with a scalar
    z = -0.5
    print(x+z)



    a = StateElement(
        values = numpy.array([[-0.237, 0]], dtype=numpy.float32),
        spaces =
            core.space.Space([
            numpy.array([-1, -1], dtype = numpy.float32), numpy.array([1, 1], dtype = numpy.float32)
            ]),
    )
    b = StateElement(
        values = numpy.array([[0.5, .5]], dtype=numpy.float32),
        spaces =
            core.space.Space([
            numpy.array([-1, -1], dtype = numpy.float32), numpy.array([1, 1], dtype = numpy.float32)
            ]),
    )


    print(a+b)
    print(b+a)
    print(a-b)
    print(b-a)
    c = numpy.array([ 0.12823329, -0.10512559])
    print(a+c)
    print(c+a)
    print(a-c)
    print(c-a)

if _str == 'mul' or _str == 'all':
    a = StateElement(
        values = numpy.array([[-0.237, 0]], dtype=numpy.float32),
        spaces =
            core.space.Space([
            numpy.array([-1, -1], dtype = numpy.float32), numpy.array([1, 1], dtype = numpy.float32)
            ]),
    )
    b = StateElement(
        values = numpy.array([[0.5, .5]], dtype=numpy.float32),
        spaces =
            core.space.Space([
            numpy.array([-1, -1], dtype = numpy.float32), numpy.array([1, 1], dtype = numpy.float32)
            ]),
    )

    c = 1
    print(a*b)
    print(a*c)
    print(b*a)
    print(c*a)


if _str == 'matmul' or _str == 'all':
    a = StateElement(
        values = numpy.array([[-0.237, 0], [1, 1]], dtype=numpy.float32),
        spaces =
            core.space.Space([
            numpy.array([[-1, -1], [-1, -1]], dtype = numpy.float32), numpy.array([[1, 1], [1, 1]], dtype = numpy.float32)
            ]),
    )
    b = StateElement(
        values = numpy.array([[0.5, .5]], dtype=numpy.float32),
        spaces =
            core.space.Space([
            numpy.array([-1, -1], dtype = numpy.float32).reshape(-1,1), numpy.array([1, 1], dtype = numpy.float32).reshape(-1,1)
            ]),
    )

    z = numpy.ones((2,2))


    print(a @ b)
    print(z @ a)
    print(a @ z)

if _str == 'mode' or _str == 'all':
    b = StateElement(
        values = numpy.array([[3, -5]], dtype=numpy.float32),
        spaces =
            core.space.Space([
            numpy.array([-1, -1], dtype = numpy.float32).reshape(-1,1), numpy.array([1, 1], dtype = numpy.float32).reshape(-1,1)
            ]),
        clipping_mode = 'clip'
    )

    print(b)

    b = StateElement(
        values = numpy.array([[3, -5]], dtype=numpy.float32),
        spaces =
            core.space.Space([
            numpy.array([-1, -1], dtype = numpy.float32).reshape(-1,1), numpy.array([1, 1], dtype = numpy.float32).reshape(-1,1)
            ]),
        clipping_mode = 'warning'
    )

    print(b)
    try:
        b = StateElement(
            values = numpy.array([[3, -5]], dtype=numpy.float32),
            spaces =
                core.space.Space([
                numpy.array([-1, -1], dtype = numpy.float32).reshape(-1,1), numpy.array([1, 1], dtype = numpy.float32).reshape(-1,1)
                ]),
            clipping_mode = 'error'
        )
    except StateNotContainedError:
        print('returned error as expected')