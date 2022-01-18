import numpy
import itertools
import warnings


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
        Space([numpy.array([1,2,3]), numpy.array([1,2,3,4,5])], 'multidiscrete')

    :type array_bound: list(numpy.ndarray), numpy.ndarray

    :param space_type: 'continuous' or 'discrete' or 'multidiscrete'. The type of the space, see examples above.
    :type space_type: string
    :param seed: seed for the rng associated with this space, defaults to None
    :type seed: int, optional
    :param contains: 'hard' or 'soft', defaults to 'soft'. Is used when determining if an item belongs to the space. If 'soft', a more lenient comparison is used where broadcasting and viewcasting is allowed, but not typecasting.

    .. code-block:: python

        # Contains 'hard' versus 'soft'

        # ====== Discrete ======
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
        s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete", contains="hard")
        assert 1 not in s
        assert [1] not in s
        assert [[1]] not in s
        assert numpy.array(1) not in s
        assert numpy.array([1]) in s
        assert numpy.array([[1]]) not in s
        assert numpy.array([2.0]) not in s

        # ====== Multidiscrete
        s = Space(
            [
                numpy.array([1, 2, 3], dtype=numpy.int16),
                numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
            ],
            "multidiscrete",
            contains="soft",
        )
        # Soft
        assert [1, 1] in s
        assert [[1], [1]] in s
        assert [[1, 1]] in s
        assert numpy.array([1, 1]) in s
        assert numpy.array([[1], [1]]) in s
        assert numpy.array([[1, 1]]) in s

        assert numpy.array([[2], [5]]) in s
        assert numpy.array([[3], [3]]) in s

        assert numpy.array([[2.0], [5.0]]) not in s
        assert numpy.array([[2, 5]], dtype=numpy.float32) not in s
        assert numpy.array([[2], [5.0]]) not in s

        # Hard
        s = Space(
            [
                numpy.array([1, 2, 3], dtype=numpy.int16),
                numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
            ],
            "multidiscrete",
            contains="hard",
        )
        assert [1, 1] not in s
        assert [[1], [1]] not in s
        assert [[1, 1]] not in s
        assert numpy.array([1, 1]) not in s
        assert numpy.array([[1], [1]]) in s
        assert numpy.array([[1, 1]]) not in s

        assert numpy.array([[2], [5]]) in s
        assert numpy.array([[3], [3]]) in s

        assert numpy.array([[2.0], [5.0]]) not in s
        assert numpy.array([[2, 5]], dtype=numpy.float32) not in s
        assert numpy.array([[2], [5.0]]) not in s

        # ====== Continuous
        # Soft
        s = Space(
            [
                -numpy.ones((2, 2), dtype=numpy.float32),
                numpy.ones((2, 2), dtype=numpy.float32),
            ],
            "continuous",
            contains="soft",
        )
        assert [0.0, 0.0, 0.0, 0.0] in s
        assert [[0.0, 0.0], [0.0, 0.0]] in s
        assert numpy.array([0.0, 0.0, 0.0, 0.0]) in s
        assert numpy.array([[0.0, 0.0], [0.0, 0.0]]) in s

        assert 1.0 * numpy.ones((2, 2)) in s
        assert -1.0 * numpy.ones((2, 2)) in s

        assert numpy.ones((2, 2), dtype=numpy.int16) in s

        # Hard
        s = Space(
            [
                -numpy.ones((2, 2), dtype=numpy.float32),
                numpy.ones((2, 2), dtype=numpy.float32),
            ],
            "continuous",
            contains="hard",
        )
        assert [0.0, 0.0, 0.0, 0.0] not in s
        assert [[0.0, 0.0], [0.0, 0.0]] not in s
        assert numpy.array([0.0, 0.0, 0.0, 0.0]) not in s
        assert numpy.array([[0.0, 0.0], [0.0, 0.0]]) in s

        assert 1.0 * numpy.ones((2, 2)) in s
        assert -1.0 * numpy.ones((2, 2)) in s

        assert numpy.ones((2, 2), dtype=numpy.int16) in s


    """

    def __init__(
        self,
        array_bound,
        space_type,
        *args,
        dtype=None,
        seed=None,
        contains="soft",
        **kwargs
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
        if dtype is not None:
            if space_type == "discrete":
                self._array_bound = array_bound.astype(dtype)
            else:
                self._array_bound = [a.astype(dtype) for a in array_bound]
        else:
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
        if item is None:
            return False
        if self.contains == "hard":
            if not isinstance(item, numpy.ndarray):
                return False
            if item.shape != self.shape:
                return False
            if not numpy.can_cast(item.dtype, self.dtype, "same_kind"):
                return False

        elif self.contains == "soft":
            # if len(item) != len(self):
            #     return False
            if not hasattr(item, "shape"):
                item = numpy.asarray(item)
            if item.shape != self.shape:
                try:
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

        if isinstance(other, Space):
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
                result = Space(
                    self._array_bound[self.n],
                    "discrete",
                    *self.args,
                    seed=self.seed,
                    contains=self.contains,
                    **self.kwargs,
                )
            elif self.space_type == "continuous":
                if self.__iter__row:

                    result = Space(
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
                    result = Space(
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

    def __getitem__(self, key):
        """__getitem__

        .. warning::

            Indexing with an integer on continuous spaces may be counterintuitive when compared to __iter__'s behavior.
            For example, iterating on a continuous space will return spaces made of the rows of the initial space if dim[0] > 1, and columns of the initial space if dim[0] == 1. However, space[0] is equivalent to space[0,0]. So, contrary to what may be expected, next(iter(s)) != s[0] i.e. the first element returned when for-looping over a space is not s[0].

        """
        if self.space_type == "discrete":
            raise TypeError(
                "'{}' object with space_type='{}' is not subscriptable".format(
                    type(self).__name__, self.space_type
                )
            )
        if isinstance(key, int):
            if self.space_type == "continuous":
                from coopihc.space.utils import ContinuousSpaceIntIndexingWarning

                warnings.warn(
                    ContinuousSpaceIntIndexingWarning(
                        "You are indexing a continuous space with an integer index, which may be ambiguous. "
                    )
                )
                return Space(
                    [
                        numpy.atleast_2d(self.low.ravel()[key]),
                        numpy.atleast_2d(self.high.ravel()[key]),
                    ],
                    "continuous",
                    seed=self.seed,
                    contains=self.contains,
                )
            elif self.space_type == "multidiscrete":

                return Space(
                    self._array_bound[key],
                    "discrete",
                    seed=self.seed,
                    contains=self.contains,
                )
            else:
                raise NotImplementedError
        elif isinstance(key, (slice, tuple)):
            if self.space_type == "continuous":
                return Space(
                    [
                        numpy.atleast_2d(self.low[key]),
                        numpy.atleast_2d(self.high[key]),
                    ],
                    "continuous",
                    seed=self.seed,
                    contains=self.contains,
                )
            elif self.space_type == "multidiscrete":
                _array = numpy.array(self._array_bound)[key]
                if len(_array) == 1:
                    # Does not work for some reason: leads to a nested list as input for Space(, 'discrete')
                    # return self.__getitem__(key.start)
                    raise NotImplementedError
                else:

                    space_type = "multidiscrete"
                # A length one tuple will give a discrete space. This could be detected easily with len(key). However, it is harder to detect the length of a slice, since it depends on the array it is evaluated on. To account for this, we just try to use a multidiscrete space and if that fails try a discrete space instead.
                try:
                    return Space(
                        _array,
                        space_type,
                        seed=self.seed,
                        contains=self.contains,
                    )
                except:
                    return Space(
                        _array,
                        "discrete",
                        seed=self.seed,
                        contains=self.contains,
                    )
        else:
            raise NotImplementedError

    def __repr__(self):
        """__repr__"""
        if self.space_type == "continuous":
            return "Space([{}, {}], 'continuous', contains = '{}')".format(
                self.low, self.high, self.contains
            )
        elif self.space_type == "discrete":
            return "Space({}, 'discrete', contains = '{}')".format(
                self._array_bound, self.contains
            )
        elif self.space_type == "multidiscrete":
            return "Space([{}], 'multidiscrete', contains = '{}')".format(
                ",".join([str(a) for a in self._array_bound]), self.contains
            )
        else:
            return super().__repr__()

    def _flat(self):
        if self.space_type == "continuous":
            return "Continuous{}".format(self.shape)
        elif self.space_type == "discrete":
            return "Discrete({})".format(self.N)
        elif self.space_type == "multidiscrete":
            return "MultiDiscrete({})".format(self.N)
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

    def sample(self):
        """sample

        Sample a value uniformly from the space. The seed of the rng is set when initializing the Space instance.

        .. code-block:: python

             s = Space(
                numpy.array([i for i in range(1000)], dtype=numpy.int16),
                "discrete",
                contains="hard",
                seed=123,
            )
            assert s.sample() in s

        :return: a random value contained in the Space
        :rtype: numpy.ndarray
        """
        _l = []
        if self.space_type == "continuous":
            # if high and/or lows contain numpy.inf, this will lead to nans and/or infs. In that case, just sample from a centered unit Gaussian. This may trigger warnings if either the lower or upper bound is infinite but not the other since the sample may not be actually included in the space.
            return numpy.nan_to_num(
                (self.high - self.low), nan=1, posinf=1
            ) * self.rng.random(self.shape, dtype=self.dtype) + numpy.nan_to_num(
                self.low, neginf=1
            )
        elif self.space_type == "discrete":
            return numpy.array([self.rng.choice(self._array_bound).astype(self.dtype)])
        elif self.space_type == "multidiscrete":
            return numpy.array(
                [[self.rng.choice(_ab).astype(self.dtype)] for _ab in self._array_bound]
            )
        else:
            raise NotImplementedError

    def serialize(self):
        """serialize

            A representation of the Space that is JSONable.

            .. code-block:: python

                s = Space(
                    numpy.array([i for i in range(10)], dtype=numpy.int16),
                    "discrete",
                    contains="hard",
                    seed=123,
                )
                assert (
                    json.dumps(s.serialize())
                    == '{"array_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "space_type": "discrete", "seed": 123, "contains": "hard"}'
        )


            :return: JSONable Space
            :rtype: dictionnary
        """
        if self.space_type == "discrete":
            array_list = self._array_bound.tolist()
        else:
            array_list = [ab.tolist() for ab in self._array_bound]
        return {
            "array_list": array_list,
            "space_type": self.space_type,
            "seed": self.seed,
            "contains": self.contains,
        }

    @staticmethod
    def cartesian_product(*spaces):
        """cartesian_product

        Realizes the cartesian product of the spaces provided in input. For this method, continuous spaces are treated as singletons {None}.

        .. code-block:: python

            s = Space(
                numpy.array([i for i in range(3)], dtype=numpy.int16),
                "discrete",
                contains="hard",
                seed=123,
            )
            q = Space(
                [
                    numpy.array([i + 6 for i in range(2)], dtype=numpy.int16),
                    numpy.array([i + 6 for i in range(2)], dtype=numpy.int16),
                ],
                "multidiscrete",
                contains="hard",
                seed=789,
            )
            r = Space(
                [
                    -numpy.ones((2, 2), dtype=numpy.float32),
                    numpy.ones((2, 2), dtype=numpy.float32),
                ],
                "continuous",
                contains="hard",
                seed=456,
            )
            cp, shape = Space.cartesian_product(s, q, r)
            assert (
                cp
                == numpy.array(
                    [
                        [0, 6, 6, None],
                        [0, 6, 7, None],
                        [0, 7, 6, None],
                        [0, 7, 7, None],
                        [1, 6, 6, None],
                        [1, 6, 7, None],
                        [1, 7, 6, None],
                        [1, 7, 7, None],
                        [2, 6, 6, None],
                        [2, 6, 7, None],
                        [2, 7, 6, None],
                        [2, 7, 7, None],
                    ]
                )
            ).all()
            assert shape == [(1,), (2, 1), (2, 2)]


        :return: cartesian product and shape of associated spaces
        :rtype: tuple(numpy.ndarray, list(tuples))
        """
        arrays = []
        shape = []
        for space in spaces:
            shape.append(space.shape)
            if space.space_type == "discrete":
                arrays.append(space._array_bound)
            elif space.space_type == "continuous":
                arrays.append(numpy.array([None]))
            elif space.space_type == "multidiscrete":
                arrays.extend(space._array_bound)
        la = len(arrays)
        dtype = numpy.result_type(*arrays)
        arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(numpy.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la), shape
