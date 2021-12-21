import numpy
import gym
import itertools


class Space:
    """Space

    An object that defines a domain. Space is supposed to be used as an attribute to a StateElement.

    :param array_bound: the arrays that specify the domain of the space.

    .. code-block:: python

        # Example init
        # discrete space -- Space(1darray) -- the 2D array is assumed to be ordered along the columns
        Space(numpy.array([1,2,3]), 'discrete')
        # continuous space -- Space([2dlowarray, 2dhigharray])
        Space([-numpy.ones((2,2)), numpy.ones((2,2))], 'continuous')
        # multidiscrete -- Space ( [1drange, 1drange,...]) --- Arrays are assumed to be ordered
        Space([numpy.array([1,2,3]), numpy.array([1,2,3,4,5])])

    :type array_bound: list(numpy.ndarray), numpy.ndarray

    :param space_type: 'continuous' or 'discrete' or 'multidiscrete'. The type of the space, see examples above.
    :type space_type: string
    :param seed: seed for the rng associated with this space, defaults to None
    :type seed: int, optional
    :param contains: 'hard' or 'soft', defaults to 'soft'. Is used when determining if an item belongs to the space. If 'soft', a more lenient comparison is used where broadcasting and viewcasting is allowed, but not typecasting.

    .. code-block:: python

        # Soft
        s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete", contains="soft")
        assert 1 in s
        assert [1] in s
        assert [[1]] in s
        assert numpy.array(1) in s
        assert numpy.array([1]) in s
        assert numpy.array([[1]]) in s
        assert numpy.array([1.0]) not in s

        # Hard
        assert 1 not in s
        assert [1] not in s
        assert [[1]] not in s
        assert numpy.array(1) not in s
        assert numpy.array([1]) in s
        assert numpy.array([[1]]) not in s
        assert numpy.array([2.0]) not in s

    """

    def __init__(
        self, array_bound, space_type, *args, seed=None, contains="soft", **kwargs
    ):
        self.space_type = space_type
        self.seed = seed
        self.rng = numpy.random.default_rng(seed)
        self.contains = contains
        self.args = args
        self.kwargs = kwargs

        bound_shape = 1

        if space_type == "discrete":
            _array_bound = [array_bound]
        elif space_type == "multidiscrete":
            _array_bound = array_bound
        elif space_type == "continuous":
            _array_bound = array_bound
            bound_shape = 2
        else:
            raise NotImplementedError

        for bound in _array_bound:
            if len(bound.shape) != bound_shape:
                return AttributeError(
                    "Input argument   array_bound   should be of shape {} but is of shape {} ({}).".format(
                        bound_shape, len(bound.shape), bound.shape
                    )
                )

        self._array_bound = array_bound
        self._shape = None
        self._dtype = None

    def __len__(self):
        """__len__

        Returns the number of items of any value array contained in this space. Examples:

        .. code-block:: python

            s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
            assert len(s) == 1
            s = Space([numpy.array([1, 2, 3], dtype=numpy.int16), numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16)],"multidiscrete", contains="soft")
            assert len(s) == 2
            s = Space([-numpy.ones((2, 2), dtype=numpy.float32), numpy.ones((2, 2), dtype=numpy.float32)], "continuous", contains="hard")
            assert len(s) == 4


        :return: length
        :rtype: int
        """
        if self.space_type == "continuous":
            return self.shape[0] * self.shape[1]
        elif self.space_type == "multidiscrete":
            return len(self._array_bound)
        elif self.space_type == "discrete":
            return 1
        else:
            return None

    def __contains__(self, item):
        """__contains__

        To allow Python's 'in' operator to work. See examples in init w/r hard/soft

        """
        if self.contains == "hard":
            if not isinstance(item, numpy.ndarray):
                return False
            if item.shape != self.shape:
                return False
            if not numpy.can_cast(item.dtype, self.dtype, "same_kind"):
                return False

        elif self.contains == "soft":
            if not hasattr(item, "shape"):
                item = numpy.asarray(item)
            if item.shape != self.shape:
                try:
                    # Don't actually store reshaped item, just see if it works
                    item = item.reshape(self.shape)
                except ValueError:
                    return False
            if not numpy.can_cast(item.dtype, self.dtype, "same_kind"):
                return False

        if self.space_type == "continuous":
            return numpy.all(item >= self.low) and numpy.all(item <= self.high)
        if self.space_type == "discrete":
            return item in self._array_bound
        if self.space_type == "multidiscrete":
            return numpy.array(
                [item[n] in r for n, r in enumerate(self._array_bound)]
            ).all()

    def __eq__(self, other):
        """__eq__

        To allow Python's '==' operator to work. The comparison is allowed with Space objects as well as NumPy arrays. The comparison is soft, and allows broadcasting, since it is based on numpy.equal and not numpy.array_equal.

        .. code-block:: python

            # Example with discrete space
            s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
            assert s == Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
            assert s == numpy.array([1, 2, 3], dtype=numpy.int16)
            assert s == numpy.array([[1, 2, 3]], dtype=numpy.int16)
            assert s == numpy.array([[1, 2, 3]], dtype=numpy.float32)

        :param other: object to test equality with
        :type other: Space, numpy.ndarray
        :return: True or False
        :rtype: boolean
        """

        if isinstance(other, _Space):
            try:
                return all(
                    [
                        numpy.equal(_array_self, _array_other).all()
                        for _array_self, _array_other in itertools.zip_longest(
                            self._array_bound, other._array_bound, fillvalue=None
                        )
                    ]
                )
            except ValueError:
                return False

        else:

            if isinstance(other, numpy.ndarray):  # wrap in list for discrete
                other = [other]
            if self.space_type == "discrete":
                self_array_bound = [self._array_bound]
            else:
                self_array_bound = self._array_bound
            try:
                return all(
                    [
                        numpy.equal(_array_self, _array_other).all()
                        for _array_self, _array_other in itertools.zip_longest(
                            self_array_bound, other, fillvalue=None
                        )
                    ]
                )
            except ValueError:
                return False

    def __iter__(self):
        """__iter__

        To allow Python's 'iter' function.

        """
        self.n = 0
        if not self.space_type == "continuous":
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
        """__next__

        To allow iterating over the Space.

        """
        if self.n < self.max:
            if self.space_type == "discrete":
                result = self

            elif self.space_type == "multidiscrete":
                result = _Space(
                    self._array_bound[self.n],
                    "discrete",
                    *self.args,
                    seed=self.seed,
                    contains=self.contains,
                    **self.kwargs,
                )
            elif self.space_type == "continuous":
                if self.__iter__row:

                    result = _Space(
                        [
                            self.low[self.n, :].reshape(1, -1),
                            self.high[self.n, :].reshape(1, -1),
                        ],
                        "continuous",
                        *self.args,
                        seed=self.seed,
                        contains=self.contains,
                        **self.kwargs,
                    )
                else:
                    result = _Space(
                        [
                            self.low[:, self.n].reshape(-1, 1),
                            self.high[:, self.n].reshape(-1, 1),
                        ],
                        "continuous",
                        *self.args,
                        seed=self.seed,
                        contains=self.contains,
                        **self.kwargs,
                    )
            self.n += 1
            return result
        else:
            raise StopIteration

    def __repr__(self):
        """__repr__"""
        if self.space_type == "continuous":
            return "Space[Continuous({}), {}]".format(self.shape, self.dtype)
        elif self.space_type == "discrete":
            return "Space[Discrete({}), {}]".format(self.N, self.dtype)
        elif self.space_type == "multidiscrete":
            return "Space[MultiDiscrete({}), {}]".format(self.N, self.dtype)
        else:
            return super().__repr__()

    @property
    def low(self):
        """low

        Return the lower bound for the Space. Outputs varies on the type of the space:

        .. code-block:: python

            s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
            assert s.low == 1
            s = Space([numpy.array([1, 2, 3], dtype=numpy.int16), numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16)],"multidiscrete", contains="soft")
            assert (s.low == numpy.array([1, 1])).all()
            s = Space([-numpy.ones((2, 2), dtype=numpy.float32), numpy.ones((2, 2), dtype=numpy.float32)], "continuous", contains="hard")
            assert (s.low == -numpy.ones((2, 2), dtype=numpy.float32)).all()


        :return: lower bound
        :rtype: int, numpy.ndarray
        """
        if self.space_type == "continuous":
            low = self._array_bound[0]
        elif self.space_type == "multidiscrete":
            low = numpy.asarray([numpy.min(v) for v in self._array_bound])
        else:
            low = self._array_bound[0]
        return low

    @property
    def high(self):
        """high

        Return the upper bound for the Space. Outputs varies on the type of the space:

        .. code-block:: python

            s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
            assert s.high == 3
            s = Space([numpy.array([1, 2, 3], dtype=numpy.int16), numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16)],"multidiscrete", contains="soft")
            assert (s.high == numpy.array([3, 5])).all()
            s = Space([-numpy.ones((2, 2), dtype=numpy.float32), numpy.ones((2, 2), dtype=numpy.float32)], "continuous", contains="hard")
            assert (s.high == numpy.ones((2, 2), dtype=numpy.float32)).all()


        :return: upper bound
        :rtype: int, numpy.ndarray
        """
        if self.space_type == "continuous":
            high = self._array_bound[-1]
        elif self.space_type == "multidiscrete":
            high = numpy.asarray([numpy.max(v) for v in self._array_bound])
        else:
            high = self._array_bound[-1]
        return high

    @property
    def N(self):
        """N

        Cardinality of the space. Returns None for continuous spaces:

        .. code-block:: python

            s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
            assert s.N == 3
            s = Space([numpy.array([1, 2, 3], dtype=numpy.int16), numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16)],"multidiscrete", contains="soft")
            assert s.N == [3, 5]
            s = Space([-numpy.ones((2, 2), dtype=numpy.float32), numpy.ones((2, 2), dtype=numpy.float32)], "continuous", contains="hard")
            assert s.N == None

        :return: Cardinality of the spaces
        :rtype: int, list, None
        """
        if self.space_type == "continuous":
            return None
        elif self.dtype == object:
            return None
        elif self.space_type == "discrete":
            return max(self._array_bound.shape)
        else:
            return [len(a) for a in self._array_bound]

    @property
    def dtype(self):
        """dtype

        Returns dtype of the space. In case of conflict resort to numpy.find_common_type

        :return: numpy dtype
        :rtype: numpy.dtype
        """
        if self._dtype is None:
            if self.space_type == "discrete":
                self._dtype = self._array_bound.dtype
            else:
                self._dtype = numpy.find_common_type(
                    [v.dtype for v in self._array_bound], []
                )
        return self._dtype

    @property
    def shape(self):
        """shape

        Returns the shape of any value array that belongs to the space.

        .. code-block:: python

            s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
            assert s.shape == (1,)
            s = Space([numpy.array([1, 2, 3], dtype=numpy.int16), numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16)],"multidiscrete", contains="soft")
            assert s.shape == (2, 1)
            s = Space([-numpy.ones((2, 2), dtype=numpy.float32), numpy.ones((2, 2), dtype=numpy.float32)], "continuous", contains="hard")
            assert s.shape == (2, 2)

        :return: value array shape
        :rtype: tuple
        """
        if self._shape is None:
            if self.space_type == "continuous":
                self._shape = self.low.shape
            elif self.space_type == "multidiscrete":
                self._shape = (len(self), 1)
            elif self.space_type == "discrete":
                self._shape = (1,)
            else:
                return None

        return self._shape


# class Space:
#     """Class used to define in which domain values of state elements live.


#     Input should be given as a list of arrays:

#         * In case the data is continuous, then provide [low, high], where low and high are arrays that define the values of the lower and upper bounds
#         * In case the data is discrete, provide [a1, a2, a3, ...], where the ai's are different ranges corresponding to different subspaces (multidiscrete space)


#     If data is discrete it is stored as a column (N,1) array.

#     :param [numpy.array] array_list: A list of NumPy arrays that specifies the ranges of the Space.
#     :param *args: For future use.
#     :param **kwargs: For future use `**kwargs`.
#     """

#     def __init__(self, array_list, *args, seed=None, **kwargs):
#         self._cflag = None
#         self.seed = seed
#         self.rng = numpy.random.default_rng(seed)

#         # Deal with variable format input
#         if isinstance(array_list, numpy.ndarray):
#             pass
#         else:
#             for _array in array_list:
#                 if not isinstance(_array, numpy.ndarray):
#                     raise AttributeError(
#                         "Input argument array_list must be or must inherit from numpy.ndarray instance."
#                     )
#         self._array = [numpy.atleast_2d(_a) for _a in array_list]

#         self._range = None
#         self._shape = None
#         self._dtype = None

#     def __len__(self):
#         return len(self._array)

#     def __repr__(self):
#         if self.continuous:
#             return "Space[Continuous({}), {}]".format(self.shape, self.dtype)
#         else:
#             return "Space[Discrete({}), {}]".format(
#                 [max(r.shape) for r in self.range], self.dtype
#             )

#     def __contains__(self, item):
#         if item.shape != self.shape:
#             try:
#                 # Don't actually store reshaped item, just see if it works
#                 item.reshape(self.shape)
#             except ValueError:
#                 return False
#         if not numpy.can_cast(item.dtype, self.dtype, "same_kind"):
#             return False
#         if self.continuous:
#             return numpy.all(item >= self.low) and numpy.all(item <= self.high)
#         else:
#             return numpy.array([item[n] in r for n, r in enumerate(self.range)]).all()

#     def __eq__(self, other):
#         # Compare dtype
#         if self.dtype != other.dtype:
#             return False
#         # Compare shape
#         if self.shape != other.shape:
#             return False
#         # Compare ranges
#         ranges_are_equal = numpy.array(
#             [
#                 numpy.array_equal(self_range_element, other_range_element)
#                 for (self_range_element, other_range_element) in zip(
#                     self.range, other.range
#                 )
#             ]
#         )
#         return ranges_are_equal.all()

#     def __iter__(self):
#         self.n = 0
#         if not self.continuous:
#             self.max = len(self)
#         else:
#             if self.shape[0] != 1:
#                 self.max = self.shape[0]
#                 self.__iter__row = True
#             elif self.shape[0] == 1 and self.shape[1] != 1:
#                 self.max = self.shape[1]
#                 self.__iter__row = False
#             else:
#                 self.max = 1
#                 self.__iter__row = True
#         return self

#     def __next__(self):

#         if self.n < self.max:
#             if not self.continuous:
#                 result = Space([self._array[self.n]])
#             else:
#                 if self.__iter__row:
#                     result = Space([self.low[self.n, :], self.high[self.n, :]])
#                 else:
#                     result = Space([self.low[:, self.n], self.high[:, self.n]])
#             self.n += 1
#             return result
#         else:
#             raise StopIteration

#     def __getitem__(self, key):
#         if isinstance(key, int):
#             if self.continuous:
#                 return self
#             else:
#                 return Space([self._array[key]])
#         elif isinstance(key, slice):
#             raise NotImplementedError
#         else:
#             raise TypeError("Index must be int, not {}".format(type(key).__name__))

#     # might be able to delete that one
#     @property
#     def range(self):
#         """If the space is continuous, returns low and high arrays, after having checked that they have the same shape. If the space is discrete, returns the list of possible values. The output is reshaped to 2d arrays.

#         :return: the 2d-reshaped ranges
#         :rtype: numpy.ndarray

#         """

#         if self._range is None:
#             if self.continuous and (not self._array[0].shape == self._array[1].shape):
#                 return AttributeError(
#                     "The low {} and high {} ranges don't have matching shapes".format(
#                         self._array[0], self._array[1]
#                     )
#                 )
#             self._range = [
#                 numpy.atleast_2d(_a) for _a in self._array
#             ]  # verify if not just self._range = self._array
#         return self._range

#     @property
#     def low(self):
#         """Return the lower end of the range. For continuous simply return low, for discrete return smallest value or the range array.

#         :return: The lower end of the range
#         :rtype: numpy.array

#         """
#         if self.continuous:
#             low = self.range[0]
#         else:
#             low = []
#             for r in self.range:
#                 r = r.squeeze()
#                 try:
#                     low.append(min(r))
#                 except TypeError:
#                     low.append(r)
#         return low

#     @property
#     def high(self):
#         """Return the higher end of the range, see low.

#         :return: The higher end of the range
#         :rtype: numpy.array

#         """
#         if self.continuous:
#             high = self.range[1]
#         else:

#             high = []
#             for r in self.range:
#                 r = r.squeeze()
#                 try:
#                     high.append(max(r))
#                 except TypeError:
#                     high.append(r)
#         return high

#     @property
#     def N(self):
#         """Returns the cardinality of the set (space) --- Only useful for 1d discrete spaces.

#         :return: Description of returned object.
#         :rtype: type

#         """
#         if self.continuous:
#             return None
#         elif self.dtype == object:
#             return None
#         else:
#             if len(self) > 1:
#                 return None

#             else:
#                 return len(self.range[0].squeeze())

#     @property
#     def shape(self):
#         """Returns the shape of the space, discrete(N) spaces are cast to (N,1).

#         :return: Shape of the space
#         :rtype: tuple

#         """
#         if self._shape is None:
#             if not self.continuous:
#                 self._shape = (len(self), 1)
#             else:
#                 self._shape = self.low.shape
#         return self._shape

#     @property
#     def dtype(self):
#         """Returns the dtype of the space. If data is in several types, will convert to the common type.

#         :return: dtype of the data
#         :rtype: numpy.dtype

#         """
#         if self._dtype is None:
#             if len(self._array) == 1:
#                 self._dtype = self._array[0].dtype
#             else:
#                 self._dtype = numpy.find_common_type([v.dtype for v in self._array], [])
#         return self._dtype

#     @property
#     def continuous(self):
#         """Whether the space is continuous or not. Based on the dtype of the provided data.

#         :return: is continuous.
#         :rtype: boolean

#         """

#         if self._cflag is None:
#             self._cflag = numpy.issubdtype(self.dtype, numpy.inexact)
#         return self._cflag

#     def sample(self):
#         """Uniforly samples from the space.

#         :return: random value in the space.
#         :rtype: numpy.array

#         """
#         _l = []
#         if self.continuous:
#             return (self.high - self.low) * self.rng.random(
#                 self.shape, dtype=self.dtype
#             ) + self.low
#         else:
#             # The conditional check for __iter__ is here to deal with single value spaces. Will not work if that value happens to be a string, but that is okay.
#             return numpy.array(
#                 [
#                     self.rng.choice(r.squeeze(), replace=True).astype(self.dtype)
#                     if hasattr(r.squeeze().tolist(), "__iter__")
#                     else r
#                     for r in self.range
#                 ]
#             ).reshape(self.shape)

#     def serialize(self):
#         """Call this to generate a dict representation of Space.

#         :return: dictionary representation of a Space object
#         :rtype: dict
#         """

#         return {"array_list": self._array}

#     def convert_to_gym(self):

#         # == If continuous
#         if numpy.issubdtype(self.dtype, numpy.inexact):
#             return [gym.spaces.Box(self.low, self.high)]
#         else:
#             ret_space = []
#             for sp in self:
#                 ret_space.append(gym.spaces.Discrete(sp.N))
#         return ret_space
