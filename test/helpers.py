from coopihc.space import StateElement
from coopihc.helpers import sort_two_lists, isdefined, flatten
import gym
import numpy
import sys

_str = sys.argv[1]


if _str == "sort_two_lists" or _str == "all":
    list1 = [1, 0, 2, 3, -1]
    x = StateElement(
        values=[
            numpy.array([1]).reshape(
                1,
            ),
            2,
            3,
        ],
        spaces=[
            gym.spaces.Box(-1, 1, shape=(1,)),
            gym.spaces.Discrete(3),
            gym.spaces.Discrete(6),
        ],
        possible_values=[[None], [None], [-6, -5, -4, -3, -2, -1]],
    )
    list2 = [x, x, x, x, x]
    sortedlist1, sortedlist2 = sort_two_lists(list1, list2, lambda pair: pair[0])

if _str == "isdefined" or _str == "all":
    a = [1, 2, None]
    b = None
    c = [1, 2, 3, 4]
    d = [[1, 2], [3, 4]]
    e = [[1, 2], [None], [4, 5]]
    print(isdefined(a), isdefined(b), isdefined(c), isdefined(d), isdefined(e))


if _str == "flatten" or _str == "all":
    u = [1, 2, 3]
    v = [1, 2, 3, u]
    w = [u, [u], [u, [u]]]
    x = 0
    print(flatten(u), flatten(v), flatten(w), flatten(x))
