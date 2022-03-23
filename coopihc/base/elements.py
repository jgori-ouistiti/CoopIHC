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
def lin_space(start, stop, num=50, endpoint=True, dtype=numpy.int64):
    # lin_space(num=50, start=0, stop=None, endpoint=False, dtype=numpy.int64):
    """Linearly spaced discrete space.

    Wrap numpy's linspace to produce a space that is compatible with COOPIHC. Parameters of this function are defined in https://numpy.org/doc/stable/reference/generated/numpy.linspace.html


    :return: _description_
    :rtype: _type_
    """
    if stop is None:
        stop = num + start
    return Space(
        array=numpy.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype)
    )


def integer_set(N, **kwargs):
    """{0, 1, ... , N-1} Set

    Wrapper around lin_space.

    :param N: cardinality of set
    :type N: int
    :return: Integer Set
    :rtype: CatSet
    """
    return lin_space(0, None, num=N, endpoint=False, **kwargs)


def integer_space(N=None, start=0, stop=None, **kwargs):
    """[0, 1, ... , N-1, N]

    Wrapper around box_space

    :param N: upper bound of discrete interval
    :type N: integer
    :return: Integer Space
    :rtype: Numeric
    """
    dtype = kwargs.get("dtype", None)
    if dtype is None:
        dtype = numpy.dtype("int64")
    if stop is None:
        if N is None:
            N = numpy.iinfo(dtype).max
        stop = N + start - 1

    return box_space(
        low=numpy.array(start),
        high=numpy.array(stop),
        **kwargs,
    )


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


def discrete_array_element(N=None, shape=None, init=0, low=None, high=None, **kwargs):
    if shape is None:
        if isinstance(init, numpy.ndarray):
            shape = init.shape
        else:
            init = numpy.asarray(init)
            shape = init.shape

    elif isinstance(shape, tuple):
        if isinstance(init, numpy.ndarray):
            if init.shape == shape:
                pass
            else:
                raise ValueError(
                    "shape arg {} inconsistent with init shape {}".format(
                        shape, init.shape
                    )
                )
        else:
            init = numpy.full(shape, init)
    elif isinstance(shape, (int, float)):
        shape = (int(shape),)
        init = numpy.full(shape, init)
    else:
        raise NotImplementedError

    if low is None:
        low = numpy.full(shape, 0)

    if high is not None:
        try:
            if high.shape != shape:
                high = numpy.full(shape, high)
        except AttributeError:
            high = numpy.full(shape, high)
    else:
        if N is not None:
            high = numpy.full(shape, N)
        else:
            high = numpy.full(shape, numpy.iinfo(numpy.int16).max)

    out_of_bounds_mode = kwargs.pop("out_of_bounds_mode", "warning")
    seed = kwargs.pop("seed", None)

    return StateElement(
        init,
        Space(low=low, high=high, **kwargs),
        out_of_bounds_mode=out_of_bounds_mode,
        seed=seed,
    )


def array_element(shape=None, init=0, low=None, high=None, **kwargs):

    low = numpy.asarray(low)
    high = numpy.asarray(high)

    if shape is None:
        if low is not None:
            lshape = low.shape
        else:
            lshape = ()
        if high is not None:
            hshape = high.shape
        else:
            hshape = ()
        if isinstance(init, numpy.ndarray):
            ishape = init.shape
        else:
            init = numpy.asarray(init)
            ishape = init.shape

    elif isinstance(shape, tuple):
        if isinstance(init, numpy.ndarray):
            if init.shape == shape or init.shape == ():
                pass
            else:
                raise ValueError(
                    "shape arg {} inconsistent with init shape {}".format(
                        shape, init.shape
                    )
                )

    elif isinstance(shape, (int, float)):
        shape = (int(shape),)
    else:
        raise NotImplementedError

    shape = sorted(
        [x for x in [lshape, hshape, ishape, shape] if x is not None], key=len
    )[-1]

    if low is None:
        low = numpy.full(shape, -numpy.inf)
    elif isinstance(low, numpy.ndarray):
        low = low.reshape(shape)
    else:
        low = numpy.full(shape, low)
    if high is None:
        high = numpy.full(shape, numpy.inf)
    elif isinstance(high, numpy.ndarray):
        high = high.reshape(shape)
    else:
        high = numpy.full(shape, high)

    try:
        init.reshape(shape)
    except ValueError:
        init = numpy.full(shape, init)

    out_of_bounds_mode = kwargs.pop("out_of_bounds_mode", "warning")
    seed = kwargs.pop("seed", None)

    return StateElement(
        init,
        Space(low=low, high=high, **kwargs),
        out_of_bounds_mode=out_of_bounds_mode,
        seed=seed,
    )


def cat_element(N, init=0, **kwargs):
    out_of_bounds_mode = kwargs.pop("out_of_bounds_mode", "warning")
    seed = kwargs.pop("seed", None)
    return StateElement(
        init, integer_set(N, **kwargs), out_of_bounds_mode=out_of_bounds_mode, seed=seed
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
