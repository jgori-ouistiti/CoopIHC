import numpy
import collections

def flatten(l):
    out = []
    if isinstance(l, collections.OrderedDict):
        l = list(l.values())
    for item in l:
        if isinstance(item, (list, tuple)):
          out.extend(flatten(item))
        else:
            if isinstance(item, numpy.ndarray):
                out.extend(flatten(item.tolist()))
            elif isinstance(item, collections.OrderedDict):
                out.extend(flatten(list(item.values())))
            else:
                out.append(item)
    return out


def sort_two_lists(list1, list2):
    sortedlist1, sortedlist2 = [list(u) for u in zip(*sorted(zip(list1, list2)))]
    return sortedlist1, sortedlist2
