from xml.dom.minidom import Attr
from coopihc.base.Space import Space
from coopihc.base.StateElement import StateElement
from coopihc.base.State import State

import numpy


# def _numpy_max_info(dtype):
#     if numpy.issubdtype(dtype, numpy.integer):
#         return numpy.iinfo(dtype).max
#     else:
#         return numpy.finfo(dtype).max


# ======================== Space Shortcuts ========================
# def lin_space(start, stop, num=50, endpoint=True, dtype=numpy.int64, seed=None, **kwargs):
#     # lin_space(num=50, start=0, stop=None, endpoint=False, dtype=numpy.int64):
#     """Linearly spaced discrete space.

#     Wrap numpy's linspace to produce a space that is compatible with COOPIHC. Parameters of this function are defined in https://numpy.org/doc/stable/reference/generated/numpy.linspace.html


#     :return: _description_
#     :rtype: _type_
#     """
#     if stop is None:
#         stop = num + start
#     return Space(
#         array=numpy.linspace(
#             start, stop, num=num, endpoint=endpoint, dtype=dtype
#         ),
#         seed=seed,
#         **kwargs
#     )


def integer_set(N, dtype=numpy.int64, **kwargs):
    """{0, 1, ... , N-1} Set

    Wrapper around lin_space.

    :param N: cardinality of set
    :type N: int
    :return: Integer Set
    :rtype: CatSet
    """
    return Space(
        array=numpy.linspace(0, N, num=N, endpoint=False), dtype=dtype, **kwargs
    )


def integer_space(N=None, start=0, stop=None, dtype=numpy.int64):
    """[0, 1, ... , N-1, N]

    Wrapper around box_space

    :param N: upper bound of discrete interval
    :type N: integer
    :return: Integer Space
    :rtype: Numeric
    """

    if stop is None:
        if N is None:
            N = numpy.iinfo(dtype).max
        stop = N + start - 1

    return box_space(low=numpy.array(start), high=numpy.array(stop), dtype=dtype)


def box_space(high=numpy.ones((1, 1)), low=None, **kwargs):
    """[low, high] Numeric

    _extended_summary_

    :param high: _description_, defaults to numpy.ones((1, 1))
    :type high: _type_, optional
    :param low: _description_, defaults to None
    :type low: _type_, optional
    :return: _description_
    :rtype: _type_
    """
    if low is None:
        low = -high
    return Space(low=low, high=high, **kwargs)


# ======================== StateElement Shortcuts ========================


def _get_shape_from_objects(shape, *obj):
    _shape = [shape]
    for _o in obj:
        try:
            _shape.append(_o.shape)
        except AttributeError:  # has no shape
            _shape.append(numpy.asarray(_o).shape)

    return sorted([x for x in _shape if x is not None], key=len)[-1]


def _set_shape_object(shape, obj, default_value):
    if obj is None:
        obj = numpy.full(shape, default_value)
    else:
        obj = numpy.asarray(obj)
        try:
            obj = obj.reshape(shape)
        except ValueError:
            obj = numpy.full(shape, obj)
    return obj


def array_element(
    shape=None, init=None, low=None, high=None, out_of_bounds_mode="warning", **kwargs
):
    shape = _get_shape_from_objects(shape, init, low, high)
    init = _set_shape_object(shape, init, 0)
    low = _set_shape_object(shape, low, -numpy.inf)
    high = _set_shape_object(shape, high, numpy.inf)

    return StateElement(
        init,
        Space(low=low, high=high, **kwargs),
        out_of_bounds_mode=out_of_bounds_mode,
    )


def discrete_array_element(
    N=None, shape=None, init=0, low=None, high=None, dtype=None, **kwargs
):
    if dtype is None:
        dtype = numpy.int64
    if N is None:
        return array_element(
            shape=shape, init=init, low=low, high=high, dtype=dtype, **kwargs
        )
    else:
        if low is None:
            low = 0
        if high is None:
            high = N - 1 + low

        return array_element(
            shape=shape, init=init, low=low, high=high, dtype=dtype, **kwargs
        )


def cat_element(N, init=0, out_of_bounds_mode="warning", **kwargs):
    return StateElement(
        init, integer_set(N, **kwargs), out_of_bounds_mode=out_of_bounds_mode
    )


def example_game_state():
    return State(
        game_info=State(
            turn_index=cat_element(
                N=4, init=0, out_of_bounds_mode="raw", dtype=numpy.int8
            ),
            round_index=discrete_array_element(init=0, out_of_bounds_mode="raw"),
        ),
        task_state=State(
            position=discrete_array_element(N=4, init=2, out_of_bounds_mode="clip"),
            targets=discrete_array_element(
                init=numpy.array([0, 1]),
                low=numpy.array([0, 0]),
                high=numpy.array([3, 3]),
            ),
        ),
        user_state=State(goal=discrete_array_element(N=4)),
        assistant_state=State(
            beliefs=array_element(
                init=numpy.array([1 / 8 for i in range(8)]),
                low=numpy.zeros((8,)),
                high=numpy.ones((8,)),
            )
        ),
        user_action=State(action=discrete_array_element(low=-1, high=1)),
        assistant_action=State(
            action=cat_element(4, init=2, out_of_bounds_mode="error")
        ),
    )
