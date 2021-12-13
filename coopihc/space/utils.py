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