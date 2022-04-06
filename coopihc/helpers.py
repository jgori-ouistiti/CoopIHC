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
