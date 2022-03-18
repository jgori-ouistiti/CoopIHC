from coopihc.base.Space import Space
from coopihc.base.StateElement import StateElement

import numpy


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


def integer_space(N, **kwargs):
    """[0, 1, ... , N-1, N]

    Wrapper around box_space

    :param N: upper bound of discrete interval
    :type N: integer
    :return: Integer Space
    :rtype: Numeric
    """
    return box_space(
        low=numpy.array(0, dtype=numpy.int64),
        high=numpy.array(N, dtype=numpy.int64),
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


def discrete_array_element(N, shape=None, init=0, low=None, **kwargs):
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

    high = numpy.full(shape, N)

    out_of_bounds_mode = kwargs.pop("out_of_bounds_mode", "warning")

    return StateElement(
        init, Space(low=low, high=high, **kwargs), out_of_bounds_mode=out_of_bounds_mode
    )


def array_element(shape=None, init=0, low=None, high=None, **kwargs):
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
        low = numpy.full(init.shape, -numpy.inf)
    elif isinstance(low, numpy.ndarray):
        low = low.reshape(shape)
    else:
        low = numpy.full(shape, low)
    if high is None:
        high = numpy.full(init.shape, numpy.inf)
    elif isinstance(high, numpy.ndarray):
        high = high.reshape(shape)
    else:
        high = numpy.full(shape, high)

    out_of_bounds_mode = kwargs.pop("out_of_bounds_mode", "warning")

    return StateElement(
        init, Space(low=low, high=high, **kwargs), out_of_bounds_mode=out_of_bounds_mode
    )


def cat_element(N, init=0, **kwargs):
    out_of_bounds_mode = kwargs.pop("out_of_bounds_mode", "warning")
    return StateElement(
        init, integer_set(N, **kwargs), out_of_bounds_mode=out_of_bounds_mode
    )
