from coopihc.space.Space import Space
import numpy


def remove_prefix(text, prefix):
    # from Python 3.9 use str.removeprefix() directly
    # copied from https://stackoverflow.com/questions/16891340/remove-a-prefix-from-a-string
    return text[text.startswith(prefix) and len(prefix) :]


class StateNotContainedWarning(Warning):
    """Warning raised when the value is not contained in the space."""

    __module__ = Warning.__module__


class SpaceLengthError(Exception):
    """Error raised when the space length does not match the value length."""

    __module__ = Exception.__module__


class StateNotContainedError(Exception):
    """Error raised when the value is not contained in the space."""

    __module__ = Exception.__module__


class SpacesNotIdenticalError(Exception):
    """Error raised when the value is not contained in the space."""

    __module__ = Exception.__module__


class NotASpaceError(Exception):
    """Error raised when the object is not a space."""

    __module__ = Exception.__module__


def discrete_space(possible_values, dtype=numpy.int16):
    """discrete_space

    Shortcut to generate a discrete Space object

    :param possible_values: possible values for the Space
    :type possible_values: numpy array_like
    :param dtype: type of the data, defaults to numpy.int16
    :type dtype: numpy dtype, optional
    :return: an initialized `Space<coopihc.space.Space.Space>` object
    :rtype: `Space<coopihc.space.Space.Space>`
    """
    return Space([numpy.array([possible_values], dtype=dtype)])


def continuous_space(low, high, dtype=numpy.float32):
    """continuous_space

    Shortcut to generate a continuous Space object

    :param low: lower bound
    :type low: numpy.ndarray
    :param high: upper bound
    :type high: numpy.ndarray
    :param dtype: type of the data, defaults to numpy.int16
    :type dtype: numpy dtype, optional
    :return: an initialized `Space<coopihc.space.Space.Space>` object
    :rtype: `Space<coopihc.space.Space.Space>`
    """
    return Space([low.astype(dtype), high.astype(dtype)])


def multidiscrete_space(iterable_possible_values, dtype=numpy.int16):
    """multidiscrete_space

    Shortcut to generate a multidiscrete_space Space object

    :param iterable_possible_values: list of possible values for the Space
    :type iterable_possible_values: twice iterable numpy array_like
    """
    return Space([numpy.array(i) for i in iterable_possible_values], dtype=dtype)
