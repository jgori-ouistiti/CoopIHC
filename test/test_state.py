from core.space import StateElement, State
import gym
import numpy
import sys

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
    x = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    for _x in x.cartesian_product():
        print(_x)

if _str == 'repr' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    y = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s = State()
    s['element1'] = x
    s['element2'] = y

    xx = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s2 = State()
    s2['element1'] = xx
    S = State()
    S['substate1'] = s
    S['substate2'] = s2


if _str == 'flat' or _str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    y = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s = State()
    s['element1'] = x
    s['element2'] = y

    xx = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    s2 = State()
    s2['element1'] = xx
    S = State()
    S['substate1'] = s
    S['substate2'] = s2


if _str == 'len' or str == 'all':
    x = StateElement(values = [numpy.array([1]).reshape(1,),2,3], spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)], possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    len(x)
