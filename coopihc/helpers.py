import numpy
import collections


def hard_flatten(l):
    # hack for circular import
    from coopihc.space.Space import Space

    out = []
    # tentative
    # if isinstance(l, (Space)):
    #     l = l.spaces
    if isinstance(l, (collections.OrderedDict, dict)):
        l = list(l.values())
    for item in l:
        if isinstance(item, (list, tuple)):
            out.extend(hard_flatten(item))
        else:
            if isinstance(item, numpy.ndarray):
                out.extend(hard_flatten(item.tolist()))
            elif isinstance(item, collections.OrderedDict):
                out.extend(hard_flatten(list(item.values())))
            # tentative
            # elif isinstance(item, (Space)):
            #     out.extend(hard_flatten(item.spaces))
            else:
                out.append(item)
    return out


def flatten(l):
    out = []
    try:
        for item in l:
            if isinstance(item, (list, tuple)):
                out.extend(flatten(item))
            else:
                out.append(item)
    except TypeError:
        return flatten([l])
    return out


def sort_two_lists(list1, list2, *args, **kwargs):
    try:
        key = args[0]
        sortedlist1, sortedlist2 = [
            list(u) for u in zip(*sorted(zip(list1, list2), key=key, **kwargs))
        ]
    except IndexError:
        sortedlist1, sortedlist2 = [
            list(u) for u in zip(*sorted(zip(list1, list2), **kwargs))
        ]

    return sortedlist1, sortedlist2


# def isdefined(obj):
#     if None not in flatten(obj):
#         return True
#     return False
