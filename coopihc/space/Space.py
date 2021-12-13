import numpy


class Space:
    """Class used to define in which domain values of state elements live.


    Input should be given as a list of arrays:

        * In case the data is continuous, then provide [low, high], where low and high are arrays that define the values of the lower and upper bounds
        * In case the data is discrete, provide [a1, a2, a3, ...], where the ai's are different ranges corresponding to different subspaces (multidiscrete space)


    If data is discrete it is stored as a column (N,1) array.

    :param [numpy.array] array_list: A list of NumPy arrays that specifies the ranges of the Space.
    :param \*args: For future use.
    :param \*\*kwargs: For future use `\*\*kwargs`.
    """

    def __init__(self, array_list, *args, seed=None, **kwargs):
        self._cflag = None
        self.rng = numpy.random.default_rng(seed)

        # Deal with variable format input
        if isinstance(array_list, numpy.ndarray):
            pass
        else:
            for _array in array_list:
                if not isinstance(_array, numpy.ndarray):
                    raise AttributeError(
                        "Input argument array_list must be or must inherit from numpy.ndarray instance."
                    )
        self._array = array_list

        self._range = None
        self._shape = None
        self._dtype = None

    def __len__(self):
        return len(self._array)

    def __repr__(self):
        if self.continuous:
            return "Space[Continuous({}), {}]".format(self.shape, self.dtype)
        else:
            return "Space[Discrete({}), {}]".format(
                [max(r.shape) for r in self.range], self.dtype
            )

    def __contains__(self, item):
        if item.shape != self.shape:
            try:
                # Don't actually store reshaped item, just see if it works
                item.reshape(self.shape)
            except ValueError:
                return False
        if not numpy.can_cast(item.dtype, self.dtype, "same_kind"):
            return False
        if self.continuous:
            return numpy.all(item >= self.low) and numpy.all(item <= self.high)
        else:
            return numpy.array([item[n] in r for n, r in enumerate(self.range)]).all()

    def __eq__(self, other):
        if self.dtype != other.dtype:
            return False
        if self.shape != other.shape:
            return False
        for _sr, _or in zip(self.range, other.range):
            condition = _sr != _or
            try:
                if condition:
                    return False
            except ValueError:
                if condition.any():
                    return False
        return True

    def __iter__(self):
        self.n = 0
        if not self.continuous:
            self.max = len(self)
        else:
            if self.shape[0] != 1:
                self.max = self.shape[0]
                self.__iter__row = True
            elif self.shape[0] == 1 and self.shape[1] != 1:
                self.max = self.shape[1]
                self.__iter__row = False
            else:
                self.max = 1
                self.__iter__row = True
        return self

    def __next__(self):

        if self.n < self.max:
            if not self.continuous:
                result = Space([self._array[self.n]])
            else:
                if self.__iter__row:
                    result = Space([self.low[self.n, :], self.high[self.n, :]])
                else:
                    result = Space([self.low[:, self.n], self.high[:, self.n]])
            self.n += 1
            return result
        else:
            raise StopIteration

    @property
    def range(self):
        """If the space is continuous, returns low and high arrays, after having checked that they have the same shape. If the space is discrete, returns the list of possible values. The output is reshaped to 2d arrays.

        :return: the 2d-reshaped ranges
        :rtype: list

        """

        if self._range is None:
            if self.continuous:
                if not self._array[0].shape == self._array[1].shape:
                    return AttributeError(
                        "The low {} and high {} ranges don't have matching shapes".format(
                            self._array[0], self._array[1]
                        )
                    )
            self._range = [numpy.atleast_2d(_a) for _a in self._array]
        return self._range

    @property
    def low(self):
        """Return the lower end of the range. For continuous simply return low, for discrete return smallest value or the range array.

        :return: The lower end of the range
        :rtype: numpy.array

        """
        if self.continuous:
            low = self.range[0]
        else:
            low = []
            for r in self.range:
                r = r.squeeze()
                try:
                    low.append(min(r))
                except TypeError:
                    low.append(r)
        return low

    @property
    def high(self):
        """Return the higher end of the range, see low.

        :return: The higher end of the range
        :rtype: numpy.array

        """
        if self.continuous:
            high = self.range[1]
        else:

            high = []
            for r in self.range:
                r = r.squeeze()
                try:
                    high.append(max(r))
                except TypeError:
                    high.append(r)
        return high

    @property
    def N(self):
        """Returns the cardinality of the set (space) --- Only useful for 1d discrete spaces.

        :return: Description of returned object.
        :rtype: type

        """
        if self.continuous:
            return None
        elif self.dtype == numpy.object:
            return None
        else:
            if len(self) > 1:
                return None

            else:
                return len(self.range[0].squeeze())

    @property
    def shape(self):
        """Returns the shape of the space, discrete(N) spaces are cast to (N,1).

        :return: Shape of the space
        :rtype: tuple

        """
        if self._shape is None:
            if not self.continuous:
                self._shape = (len(self), 1)
            else:
                self._shape = self.low.shape
        return self._shape

    @property
    def dtype(self):
        """Returns the dtype of the space. If data is in several types, will convert to the common type.

        :return: dtype of the data
        :rtype: numpy.dtype

        """
        if self._dtype is None:
            if len(self._array) == 1:
                self._dtype = self._array[0].dtype
            else:
                self._dtype = numpy.find_common_type([v.dtype for v in self._array], [])
        return self._dtype

    @property
    def continuous(self):
        """Whether the space is continuous or not. Based on the dtype of the provided data.

        :return: is continuous.
        :rtype: boolean

        """

        if self._cflag is None:
            self._cflag = numpy.issubdtype(self.dtype, numpy.inexact)
        return self._cflag

    def sample(self):
        """Uniforly samples from the space.

        :return: random value in the space.
        :rtype: numpy.array

        """
        _l = []
        if self.continuous:
            return (self.high - self.low) * self.rng.random(
                self.shape, dtype=self.dtype
            ) + self.low
        else:
            # The conditional check for __iter__ is here to deal with single value spaces. Will not work if that value happens to be a string, but that is okay.
            return numpy.array(
                [
                    self.rng.choice(r.squeeze(), replace=True).astype(self.dtype)
                    if hasattr(r.squeeze().tolist(), "__iter__")
                    else r
                    for r in self.range
                ]
            ).reshape(self.shape)

    def serialize(self):
        """Call this to generate a dict representation of Space.

        :return: dictionary representation of a Space object
        :rtype: dict
        """

        return {"array_list": self._array}
