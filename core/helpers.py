import numpy
import collections

def hard_flatten(l):
    out = []
    if isinstance(l, collections.OrderedDict):
        l = list(l.values())
    for item in l:
        if isinstance(item, (list, tuple)):
          out.extend(hard_flatten(item))
        else:
            if isinstance(item, numpy.ndarray):
                out.extend(hard_flatten(item.tolist()))
            elif isinstance(item, collections.OrderedDict):
                out.extend(hard_flatten(list(item.values())))
            else:
                out.append(item)
    return out

def flatten(l):
    out = []
    for item in l:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


def sort_two_lists(list1, list2, *args, **kwargs):
    try:
        key = args[0]
        sortedlist1, sortedlist2 = [list(u) for u in zip(*sorted(zip(list1, list2), key = key, **kwargs))]
    except IndexError:
        sortedlist1, sortedlist2 = [list(u) for u in zip(*sorted(zip(list1, list2), **kwargs))]

    return sortedlist1, sortedlist2
