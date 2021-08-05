import core
from core.space import StateElement, State, StateNotContainedError
import gym
import numpy
import sys
import copy


_str = sys.argv[1]

# -------- Correct assigment
if _str == 'correct' or _str == 'all':
    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]]
                        )

if _str == 'non-list':
    x = StateElement(   values = numpy.array([-.5]).reshape(1,),
        spaces = core.space.Box(-1,1, shape = (1,)),
        possible_values = None)

    y = StateElement(   values = 0,
        spaces = core.space.Discrete(3),
        possible_values = [1,2,3])

if _str == 'array':
    x = StateElement(
            values = [numpy.array([0,0])],
            spaces = [core.space.Box(-1,1, shape = (2,))],
            possible_values = [None]
            )
# --------- non rigorous assigment
if _str == 'non-rigorous' or _str == 'all':
    x = StateElement(   values = [1,2,3],
                        spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]]
                        )
    y = StateElement(   values = [1,2,3],
                        spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)],
                        possible_values = None)

# ---------- nested values
if _str == 'nested' or _str == 'all':
    x = StateElement(   values = [[1],[2], [3]],
                        spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])

# -------------- accessing values
if _str == 'access' or _str == 'all':
    x = StateElement(   values = [[1],[2], [3]],
                        spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    x['values']
    x['spaces']
    x['possible_values']
    x.values
    x['values'] = [1, 1, 1]


# ------ normal reset
if _str == 'reset' or _str == 'all':
    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]] )
    x.reset()

# -------- forced reset
if _str == 'forced-reset' or _str == 'all':
    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    reset_dic = {'values': [-1/2,-0,0]}
    x.reset(dic = reset_dic)

if _str == 'gethv' or _str == 'all':
    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    x.get_human_values()
    s = State()
    s['substate'] = x
    s['substate']['human_values']


    y = StateElement(   values = [None, None, None],
                        spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])

    q = State()
    q['substate'] = y
    q['substate']['human_values']

    u = StateElement(   values = [2],
                        spaces = [core.space.Discrete(5)],
                        possible_values = [None]
                        )
    print(u['human_values'][0])

if _str == 'nones' or _str == 'all':
    x = StateElement(values = None, spaces = [core.space.Discrete(2)], possible_values  = None)
    y = StateElement()
    z = StateElement(
            values = [None, None],
            spaces = [core.space.Box(low = -numpy.inf, high = numpy.inf, shape = (1,)) for i in range(2)],
            possible_values = None
             )
    z['values'] = None

if _str == 'iter' or _str == 'all':
    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    for _x in x:
        print(_x)

if _str == 'cartesianproduct' or _str == 'all':
    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    for _x in x.cartesian_product():
        print(_x)

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

if _str == 'len' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    len(x)

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

if _str == 'neg' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,)], spaces = [core.space.Box(-1,1, shape = (1,))], possible_values = [[None]], clipping_mode = 'warning')
    print(-x)

if _str == 'add' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),1,1], spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]], clipping_mode = 'warning')
    y = [-1,-1,2]
    z = -1
    print(x+y)
    print(x+z)

    a = StateElement(values = [numpy.array([0.2177283 , 0.11400087])],
                    spaces = [core.space.Box(low = -numpy.inf, high = numpy.inf, shape = (2,))],
                    possible_values = [None], clipping_mode = 'warning')

    b = numpy.array([ 0.12823329, -0.10512559])
    print(a+b)
    print(b+a)
    print(a-b)
    print(b-a)

    x = StateElement(values = [numpy.array([0.5]).reshape(1,)], spaces = [core.space.Box(-1,1, shape = (1,))], possible_values = [[None]])

    y = StateElement(values = [numpy.array([0.5]).reshape(1,)], spaces = [core.space.Box(-1,1, shape = (1,))], possible_values = [[None]])

    print(x+y)
    print(x-y)
    d = x+y
    print(d)
    d = x-y
    print(d)

if _str == 'mul' or _str == 'all':
    a = StateElement(values = [numpy.array([0.2177283 , 0.11400087])],
                    spaces = [core.space.Box(low = -100, high = 100, shape = (2,))],
                    possible_values = [None])
    b = numpy.array([ 0.12823329, -0.10512559])
    c = 1
    print(a*b)
    print(a*c)
    print(b*a)
    print(c*a)

if _str == 'cast' or _str == 'all':

    # n to Box

    for i in range(6):
        print(i)
        x = StateElement(   values = [i],
                        spaces = [core.space.Discrete(6)],
                        possible_values = [[None]])

        y = StateElement(   values = [None],
                        spaces = [core.space.Box(-1, 1, shape = (1,))],
                        possible_values = [None]
                        )
        ret = x.cast(y)
        print(y)


        y = ret.cast(x)
        print(y)

    x = StateElement(   values = [4],
            spaces = [core.space.Discrete(9)],
            possible_values = [[None]])

    y = StateElement(   values = [None],
                    spaces = [core.space.Box(-1, 1, shape = (1,))],
                    possible_values = [None]
                    )

    ret = x.cast(y)
    print(ret)
    a = StateElement(   values = [0.234],
                    spaces = [core.space.Box(-1, 1, shape = (1,))],
                    possible_values = [None]
                    )

    b = StateElement(   values = [None],
            spaces = [core.space.Discrete(9)],
            possible_values = [[None]])

    c = StateElement(   values = [None],
            spaces = [core.space.Discrete(58)],
            possible_values = [[None]])

    ret2 = a.cast(b,)
    ret3 = a.cast(c)
    print(ret2, ret3)

    # Box to Box

    x = StateElement(   values = [0],
                        spaces = [core.space.Box(-2,2, shape = (1,))],
                        possible_values = [None])

    y = StateElement(   values = [None],
                        spaces = [core.space.Box(-1, 1, shape = (1,))],
                        possible_values = [None]
                        )

    ret3 = x.cast(y)
    ret4 = x.cast(y)
    print(y)



    x = StateElement(   values = [numpy.array([1]).reshape(1,),1,1],
                        spaces = [core.space.Box(-1,1, shape = (1,)), core.space.Discrete(3), core.space.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    y = StateElement(   values = [None, None, None],
                        spaces = [core.space.Box(-2, 2, shape = (1,)), core.space.Box(-1,1, shape = (1,)), core.space.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])

    ret = x.cast(y)
    ret2 = x.cast(y)
    print(y)

    for i in range(31):
        x = StateElement(   values = [i],
                            spaces = [ core.space.Discrete(31)],
                            possible_values = [[None]])
        y = StateElement(   values = [None],
                            spaces = [ core.space.Box(-1, 1, shape = (1,))],
                            possible_values = [[None]])
        ret = x.cast(y)
        print(i, ret)

if _str == 'matmul' or _str == 'all':
    x = StateElement(   values = [numpy.ones((2,2))],
            spaces = [core.space.Box(low = -numpy.inf*numpy.ones((2,2)), high = numpy.inf*numpy.ones((2,2)), shape = (2,2) )],
            possible_values = [[None]])

    y = StateElement(   values = [numpy.ones((2,2))],
            spaces = [core.space.Box(low = -numpy.inf*numpy.ones((2,2)), high = numpy.inf*numpy.ones((2,2)), shape = (2,2) )],
            possible_values = [[None]])

    z = numpy.ones((2,2))
    w = StateElement(   values = [numpy.ones((2,1))],
            spaces = [core.space.Box(low = -numpy.inf*numpy.ones((2,1)), high = numpy.inf*numpy.ones((2,1)), shape = (2,1) )],
            possible_values = [[None]])

    print(x @ y)
    print(x @ z)
    print(z @ x)

if _str == 'mode' or _str == 'all':
    x = StateElement(   values = [3*numpy.ones((2,2))],
            spaces = [core.space.Box(low = -1, high = 1, shape = (2,2) )],
            possible_values = [[None]], clipping_mode = 'clip')
    print(x)
    x = StateElement(   values = [3*numpy.ones((2,2))],
            spaces = [core.space.Box(low = -1, high = 1, shape = (2,2) )],
            possible_values = [[None]], clipping_mode = 'warning')
    print(x)
    try:
        x = StateElement(   values = [3*numpy.ones((2,2))],
                spaces = [core.space.Box(low = -1, high = 1, shape = (2,2) )],
                possible_values = [[None]], clipping_mode = 'error')
    except StateNotContainedError:
        print('returned error as expected')


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
