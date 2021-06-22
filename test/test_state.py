from core.space import StateElement, State
import gym
import numpy
import sys
import copy

_str = sys.argv[1]

# -------- Correct assigment
if _str == 'correct' or _str == 'all':
    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]]
                        )

if _str == 'non-list':
    x = StateElement(   values = numpy.array([-.5]).reshape(1,),
        spaces = gym.spaces.Box(-1,1, shape = (1,)),
        possible_values = None)

if _str == 'array':
    x = StateElement(
            values = [numpy.array([0,0])],
            spaces = [gym.spaces.Box(-1,1, shape = (2,))],
            possible_values = [None]
            )
# --------- non rigorous assigment
if _str == 'non-rigorous' or _str == 'all':
    x = StateElement(   values = [1,2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]]
                        )
    y = StateElement(   values = [1,2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = None)

# ---------- nested values
if _str == 'nested' or _str == 'all':
    x = StateElement(   values = [[1],[2], [3]],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])

# -------------- accessing values
if _str == 'access' or _str == 'all':
    x = StateElement(   values = [[1],[2], [3]],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    x['values']
    x['spaces']
    x['possible_values']
    x.values
    x['values'] = [1, 1, 1]


# ------ normal reset
if _str == 'reset' or _str == 'all':
    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]] )
    x.reset()

# -------- forced reset
if _str == 'forced-reset' or _str == 'all':
    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    reset_dic = {'values': [-1/2,-0,0]}
    x.reset(dic = reset_dic)

if _str == 'gethv' or _str == 'all':
    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    x.get_human_values()
    s = State()
    s['substate'] = x
    s['substate']['human_values']


    y = StateElement(   values = [None, None, None],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])

    q = State()
    q['substate'] = y
    q['substate']['human_values']

if _str == 'nones' or _str == 'all':
    x = StateElement(values = None, spaces = [gym.spaces.Discrete(2)], possible_values  = None)
    y = StateElement()
    z = StateElement(
            values = [None, None],
            spaces = [gym.spaces.Box(low = -numpy.inf, high = numpy.inf, shape = (1,)) for i in range(2)],
            possible_values = None
             )
    z['values'] = None

if _str == 'iter' or _str == 'all':
    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    for _x in x:
        print(_x)

if _str == 'cartesianproduct' or _str == 'all':
    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    for _x in x.cartesian_product():
        print(_x)

if _str == 'repr' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(15)], possible_values = [[None], [None], [-6+i for i in range(15)]])
    y = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    z = StateElement(values = [numpy.array([[1/3,1/3, 1/3],[3/5,3/5,3/5], [1/7,1/8,1/9]])], spaces = [gym.spaces.Box(-1,1, shape = (3,3))], possible_values = [[None]])
    s = State()
    s['element1'] = x
    s['element2'] = y
    s['element3'] = z

    xx = StateElement(values = [numpy.array([0,0]).reshape(2,),2,3], spaces = [gym.spaces.Box(-1,1, shape = (2,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    yy = StateElement(values = [numpy.array([[0,0],[1,1]])], spaces = [gym.spaces.Box(-1,1, shape = (2,2))], possible_values = [[None]])
    s2 = State()
    s2['element1'] = yy
    s2['element2'] = xx

    S = State()
    S['substate1'] = s
    S['substate2'] = s2





if _str == 'flat' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),1,1], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    y = StateElement(values = [numpy.array([1]).reshape(1,),2,2], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s = State()
    s['element1'] = x
    s['element2'] = y

    xx = StateElement(values = [numpy.array([1]).reshape(1,),3,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(4), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s2 = State()
    s2['element1'] = xx
    S = State()
    S['substate1'] = s
    S['substate2'] = s2

    S.flat()

if _str == 'len' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    len(x)

if _str == 'filter' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),1,1], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    y = StateElement(values = [numpy.array([1]).reshape(1,),2,2], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s = State()
    s['e1'] = x
    s['e2'] = y

    xx = StateElement(values = [numpy.array([1]).reshape(1,),3,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(4), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s2 = State()
    s2['element1'] = xx
    S = State()
    S['substate1'] = s
    S['substate2'] = s2

    from collections import OrderedDict
    ordereddict = OrderedDict({ 'substate1' : OrderedDict({'e1': 0, 'e2': 0})})

    ns1 = S.filter('values', ordereddict)
    ns2 = S.filter('spaces', ordereddict)
    ns3 = S.filter('possible_values', ordereddict)
    ns4 = S.filter('human_values', ordereddict)
    ns5 = S.filter('values', S)
    ns6 = S.filter('spaces', S)
    ns7 = S.filter('possible_values', S)
    ns8 = S.filter('human_values', S)


if _str == 'copy' or _str == 'all':

    import copy
    x = StateElement(values = [numpy.array([1]).reshape(1,),1,1], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    y = StateElement(values = [numpy.array([1]).reshape(1,),2,2], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s = State()
    s['e1'] = x
    s['e2'] = y

    xx = StateElement(values = [numpy.array([1]).reshape(1,),3,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(4), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s2 = State()
    s2['element1'] = xx
    S = State()
    S['substate1'] = s
    S['substate2'] = s2

    _copy = copy.copy(S)
    _deepcopy = copy.deepcopy(S)

    S['substate1']['e1']['values'] = [0,0,0]

if _str == 'add' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),1,1], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    y = [2,2,2]
    z = 1
    print(x+y)
    print(x+z)

    a = StateElement(values = [numpy.array([0.2177283 , 0.11400087])],
                    spaces = [gym.spaces.Box(low = -1, high = 1, shape = (2,))],
                    possible_values = [None])

    b = numpy.array([ 0.12823329, -0.10512559])
    print(a+b)

if _str == 'cast' or _str == 'all':

    # n to Box

    # for i in range(6):
    #     print(i)
    #     x = StateElement(   values = [i],
    #                     spaces = [gym.spaces.Discrete(6)],
    #                     possible_values = [[None]])
    #
    #     y = StateElement(   values = [None],
    #                     spaces = [gym.spaces.Box(-1, 1, shape = (1,))],
    #                     possible_values = [None]
    #                     )
    #     ret = x.cast(y, inplace = True)
    #     print(y)
    #
    #
    #     ret = y.cast(x, inplace = False)
    #     print(ret)

    x = StateElement(   values = [4],
            spaces = [gym.spaces.Discrete(9)],
            possible_values = [[None]])

    y = StateElement(   values = [None],
                    spaces = [gym.spaces.Box(-1, 1, shape = (1,))],
                    possible_values = [None]
                    )

    ret = x.cast(y, inplace = False)
    print(ret)
    a = StateElement(   values = [0.234],
                    spaces = [gym.spaces.Box(-1, 1, shape = (1,))],
                    possible_values = [None]
                    )

    b = StateElement(   values = [None],
            spaces = [gym.spaces.Discrete(9)],
            possible_values = [[None]])

    c = StateElement(   values = [None],
            spaces = [gym.spaces.Discrete(58)],
            possible_values = [[None]])

    ret2 = a.cast(b, inplace = False)
    ret3 = a.cast(c, inplace = False)
    print(ret2, ret3)
    #
    # # Box to Box
    #
    # x = StateElement(   values = [0],
    #                     spaces = [gym.spaces.Box(-2,2, shape = (1,))],
    #                     possible_values = [None])
    #
    # y = StateElement(   values = [None],
    #                     spaces = [gym.spaces.Box(-1, 1, shape = (1,))],
    #                     possible_values = [None]
    #                     )
    #
    # ret3 = x.cast(y, inplace = False)
    # ret4 = x.cast(y, inplace = True)
    # print(y)



    # x = StateElement(   values = [numpy.array([1]).reshape(1,),1,1],
    #                     spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
    #                     possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    # y = StateElement(   values = [None, None, None],
    #                     spaces = [gym.spaces.Box(-2, 2, shape = (1,)), gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(6)],
    #                     possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    #
    # ret = x.cast(y, inplace = False)
    # ret2 = x.cast(y, inplace = True)
    # print(y)
    #
    # for i in range(31):
    #     x = StateElement(   values = [i],
    #                         spaces = [ gym.spaces.Discrete(31)],
    #                         possible_values = [[None]])
    #     y = StateElement(   values = [None],
    #                         spaces = [ gym.spaces.Box(-1, 1, shape = (1,))],
    #                         possible_values = [[None]])
    #     ret = x.cast(y, inplace = False)
    #     print(i, ret)
