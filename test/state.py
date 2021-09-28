import sys
_str = sys.argv[1]

import core
from core.space import StateElement, Space
import numpy

if _str == 'cast' or _str == 'all':

    b = StateElement(
        values = 5,
        spaces = Space(
            numpy.array([-5,-4,-3,-2,-1,0,1,2,3,4,5], dtype = numpy.int16)
        )
    )

    a = StateElement(
        values = 0,
        spaces =
            Space([
            numpy.array([-1], dtype = numpy.float32), numpy.array([1], dtype = numpy.float32)
            ]),
    )

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
            Space([
            numpy.array([-2], dtype = numpy.float32), numpy.array([1], dtype = numpy.float32)
            ]),
    )
    b = StateElement(
        values = 3.5,
        spaces =
            Space([
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
        spaces = Space(
            numpy.array([i for i in range(11)], dtype = numpy.int16)
        )
    )
    b = StateElement(
        values = 5,
        spaces = Space(
            numpy.array([-5,-4,-3,-2,-1,0,1,2,3,4,5], dtype = numpy.int16)
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


if _str == 'repr' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(15)], possible_values = [[None], [None], [-6+i for i in range(15)]])
    y = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    z = StateElement(values = [numpy.array([[1/3,1/3, 1/3],[3/5,3/5,3/5], [1/7,1/8,1/9]])], spaces = [core.space.Box(-1,1, shape = (3,3))], possible_values = [[None]])
    s = State()
    s['element1'] = x
    s['element2'] = y
    s['element3'] = z

    xx = StateElement(values = [numpy.array([0,0]).reshape(2,),2,3], spaces = [core.space.Box(-1,1, shape = (2,)), core.space.Discrete(3), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    yy = StateElement(values = [numpy.array([[0,0],[1,1]])], spaces = [core.space.Box(-1,1, shape = (2,2))], possible_values = [[None]])
    s2 = State()
    s2['element1'] = yy
    s2['element2'] = xx

    S = State()
    S['substate1'] = s
    S['substate2'] = s2





if _str == 'flat' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),1,1], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    y = StateElement(values = [numpy.array([1]).reshape(1,),2,2], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s = State()
    s['element1'] = x
    s['element2'] = y

    xx = StateElement(values = [numpy.array([1]).reshape(1,),3,3], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(4), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s2 = State()
    s2['element1'] = xx
    S = State()
    S['substate1'] = s
    S['substate2'] = s2

    S.flat()



if _str == 'filter' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),1,1], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    y = StateElement(values = [numpy.array([1]).reshape(1,),2,2], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s = State()
    s['e1'] = x
    s['e2'] = y

    xx = StateElement(values = [numpy.array([1]).reshape(1,),3,3], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(4), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s2 = State()
    s2['element1'] = xx
    S = State()
    S['substate1'] = s
    S['substate2'] = s2

    from collections import OrderedDict
    ordereddict = OrderedDict({ 'substate1' : OrderedDict({'e1': 0, 'e2': 0})})

    ns1 = S.filter('values', filterdict = ordereddict)
    ns2 = S.filter('spaces', filterdict = ordereddict)
    ns3 = S.filter('possible_values', filterdict = ordereddict)
    ns4 = S.filter('human_values', filterdict = ordereddict)
    ns5 = S.filter('values')
    ns6 = S.filter('spaces')
    ns7 = S.filter('possible_values')
    ns8 = S.filter('human_values')


if _str == 'copy' or _str == 'all':

    import copy
    x = StateElement(values = [numpy.array([1]).reshape(1,),1,1], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    y = StateElement(values = [numpy.array([1]).reshape(1,),2,2], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s = State()
    s['e1'] = x
    s['e2'] = y

    xx = StateElement(values = [numpy.array([1]).reshape(1,),3,3], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(4), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s2 = State()
    s2['element1'] = xx
    S = State()
    S['substate1'] = s
    S['substate2'] = s2

    _copy = copy.copy(S)
    _deepcopy = copy.deepcopy(S)

    S['substate1']['e1']['values'] = [0,0,0]




if _str == 'serialize' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),1,1], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    y = StateElement(values = [numpy.array([1]).reshape(1,),2,2], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s = State()
    s['one'] = x
    s['two'] = y

    xx = StateElement(values = [numpy.array([1]).reshape(1,),3,3], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(4), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s2 = State()
    s2['element_one'] = xx
    S = State()
    S['substate_one'] = s
    S['substate_two'] = s2

    print(S.serialize())
