import warnings


# ======================== Warnings ========================
class StateNotContainedWarning(Warning):
    """Warning raised when the value is not contained in the space."""

    __module__ = Warning.__module__


class NotKnownSerializationWarning(Warning):
    """Warning raised when the State tries to serialize an item which does not have a serialize method."""

    __module__ = Warning.__module__


class ContinuousSpaceIntIndexingWarning(Warning):
    """Warning raised when the State tries to serialize an item which does not have a serialize method."""

    __module__ = Warning.__module__


class NumpyFunctionNotHandledWarning(Warning):
    """Warning raised when the numpy function is not handled yet by the StateElement."""

    __module__ = Warning.__module__


class RedefiningHandledFunctionWarning(Warning):
    """Warning raised when the numpy function is already handled by the StateElement and is going to be redefined."""

    __module__ = Warning.__module__


class WrongConvertorWarning(Warning):
    """Warning raised when the value is not contained in the space."""

    __module__ = Warning.__module__


class StateElementAssignmentWarning(Warning):
    """Warning raised when trying to assign a statelement to a state with a previous stateelement, and the two spaces don't match"""

    __module__ = Warning.__module__


# ======================== Errors ========================


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


class SpaceNotSeparableError(Exception):
    """Error raised when the space can not be indexed."""

    __module__ = Exception.__module__
