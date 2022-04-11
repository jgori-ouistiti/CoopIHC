import numpy
import itertools
import warnings


class BaseSpace:
    """Base space from which other spaces inherit.

    :param seed: seed used by the rng, defaults to None
    :type seed: int, optional
    :param dtype: dtype to which the space will be coerced, defaults to None. If None, the dtype is inferred from the supplied data.
    :type dtype: numpy.dtype, optional
    :param contains: how the Python ``in`` keyword is evaluated, defaults to "numpy". If "numpy", a value is considered in the space according to numpy's ``in``
    :type contains: str, optional
    """

    def __init__(
        self,
        seed=None,
        contains="numpy",
    ):

        self.seed = seed
        self.contains = contains

        self.rng = numpy.random.default_rng(seed)
        self._shape = None
        self._spacetype = None

    @property
    def spacetype(self):
        if self._spacetype is None:
            if numpy.issubdtype(self.dtype, numpy.integer):
                return "discrete"
            elif numpy.issubdtype(self.dtype, numpy.floating):
                return "continuous"
        else:
            raise NotImplementedError


class Space:
    """_summary_

    :param low: see `Numeric<coopihc.base.Space.Numeric>`, defaults to None
    :type low: see `Numeric<coopihc.base.Space.Numeric>`, optional
    :param high: see `Numeric<coopihc.base.Space.Numeric>`, defaults to None
    :type high: see `Numeric<coopihc.base.Space.Numeric>`, optional
    :param array: see `CatSet<coopihc.base.Space.CatSet>`, defaults to None
    :type array: see `CatSet<coopihc.base.Space.CatSet>`, optional
    :param N: for future, defaults to None
    :type N: for future, optional
    :param _function: for future, defaults to None
    :type _function: for future, optional

    :return: A CoopIHC space
    :rtype: `Numeric<coopihc.base.Space.Numeric>` or `CatSet<coopihc.base.Space.CatSet>`
    """

    def __new__(
        cls,
        low=None,
        high=None,
        array=None,
        N=None,
        _function=None,
        seed=None,
        dtype=None,
        contains="numpy",
    ):
        if low is not None and high is not None:
            return Numeric(
                low=numpy.asarray(low),
                high=numpy.asarray(high),
                seed=seed,
                dtype=dtype,
                contains=contains,
            )
        if array is not None:
            return CatSet(
                array=numpy.asarray(array), seed=seed, dtype=dtype, contains=contains
            )
        if N is not None and _function is not None:
            raise NotImplementedError
        raise ValueError(
            "You have to specify either low and high, or a set, or N and function, but you provided low = {}, high = {}, set = {}, N = {}, function = {}".format(
                low, high, array, N, _function
            )
        )

    @staticmethod
    def cartesian_product(*spaces):
        """cartesian_product

        Computes the cartesian product of the spaces provided in input. For this method, continuous spaces are treated as singletons {None}.

        .. code-block:: python

            s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
            q = space(array=numpy.array([-3, -2, -1], dtype=numpy.int16))
            cp, shape = cartesian_product(s, q)
            assert (
                cp
                == numpy.array(
                    [
                        [1, -3],
                        [1, -2],
                        [1, -1],
                        [2, -3],
                        [2, -2],
                        [2, -1],
                        [3, -3],
                        [3, -2],
                        [3, -1],
                    ]
                )
            ).all()


        :return: cartesian product and shape of associated spaces
        :rtype: tuple(numpy.ndarray, list(tuples))
        """
        arrays = []
        shape = []
        for space in spaces:
            shape.append(space.shape)
            if isinstance(space, CatSet):
                arrays.append(space.array)
            elif isinstance(space, Numeric):
                if space.spacetype == "discrete":
                    arrays.append(space.array)
                else:
                    arrays.append(numpy.array([None]))

        la = len(arrays)
        dtype = numpy.result_type(*arrays)
        arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(numpy.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la), shape


class Numeric(BaseSpace):
    """Numeric

    An interval that defines the space for a StateElement.

    You can define an Numeric by specifying the lower and upper bounds:

    .. code-block:: python

        s = Numeric(
        low=-numpy.ones((2, 2), dtype=numpy.float32),
        high=numpy.ones((2, 2), dtype=numpy.float32),
        )

        assert s.dtype == numpy.float32
        assert (s.high == numpy.ones((2, 2))).all()
        assert (s.low == -numpy.ones((2, 2))).all()
        assert s.shape == (2, 2)

    You can further set the seed of the space (useful when sampling from the space), force the dtype of the space and specify how membership to the space is checked via the keyword arguments. See ``BaseSpace`` for more information.

    .. note::

        lower and upper bounds must be valid numpy objects. For example, to specify a 0-D space, you should do: ``Numeric(low = -numpy.float64(1), high = numpy.float64(1))``



    :param low: lower bound, defaults to -numpy.array([1])
    :type low: numpy.ndarray, optional
    :param high: upper bound, defaults to numpy.array([1])
    :type high: numpy.ndarray, optional
    """

    def __init__(
        self,
        low=-numpy.array([1]),
        high=numpy.array([1]),
        seed=None,
        dtype=None,
        contains="numpy",
    ):

        if dtype is not None:
            self._dtype = numpy.dtype(dtype)
        else:
            self._dtype = None

        low = numpy.asarray(low)
        high = numpy.asarray(high)
        self._N = None
        self._array = None

        if low is not None and high is not None:
            if low.shape != high.shape:
                return ValueError(
                    "Low and high must have the same shape, but low and high have shape {} and {}".format(
                        low.shape, high.shape
                    )
                )

        self.low, self.high = low, high

        super().__init__(seed=seed, contains=contains)

        # converting numpy.inf to integer is not standardized and
        # self.low = low.astype(self.dtype)
        # self.high = high.astype(self.dtype)
        # will not work
        #  Currently, it will give -2**(nbits) /2 for both numpy.inf and -numpy.inf. Hack below

        if numpy.issubdtype(self.dtype, numpy.integer):
            self.low = numpy.nan_to_num(self.low, neginf=numpy.iinfo(self.dtype).min)
            self.high = numpy.nan_to_num(self.high, posinf=numpy.iinfo(self.dtype).max)

        self.low = self.low.astype(self.dtype)
        self.high = self.high.astype(self.dtype)

    @property
    def shape(self):
        """shape

        Returns the numpy shape of the bounds.

        """
        self._shape = self.low.shape
        return self._shape

    @property
    def dtype(self):
        """dtype

        Determines the numpy dtype of data contained in the space.

        .. note::

            If you input two different integer dtypes, the result will be a numpy.float64, per https://numpy.org/doc/stable/reference/generated/numpy.common_type.html

        :return: _description_
        :rtype: _type_
        """
        if self._dtype is None:

            if self.low.dtype == self.high.dtype:
                self._dtype = numpy.dtype(self.low.dtype)
            else:
                self._dtype = numpy.dtype(numpy.common_type(self.low, self.high))

            self.low = self.low.astype(self._dtype)
            self.high = self.high.astype(self._dtype)

        return self._dtype

    @property
    def N(self):
        if self._N is None:
            if numpy.issubdtype(self.dtype, numpy.integer):
                self._N = self.high - self.low + 1

        return self._N

    @property
    def array(self):
        if self._array is None:
            if numpy.issubdtype(self.dtype, numpy.integer):
                self._array = numpy.linspace(
                    self.low, self.high, num=self.N, endpoint=True, dtype=self.dtype
                )

        return self._array

    def __iter__(self):
        """__iter__"""
        self._iter_low = iter(self.low)
        self._iter_high = iter(self.high)

        return self

    def __next__(self):
        """__next__"""
        return type(self)(
            low=next(self._iter_low),
            high=next(self._iter_high),
            seed=self.seed,
            dtype=self.dtype,
            contains=self.contains,
        )

    def __getitem__(self, key):
        """__getitem__

        Numpy Indexing is valid:

        .. code-block:: python

            s = Numeric(
                low=-numpy.ones((2, 2), dtype=numpy.float32),
                high=numpy.ones((2, 2), dtype=numpy.float32),
            )
            assert s[0] == Numeric(
                low=-numpy.ones((2,), dtype=numpy.float32),
                high=numpy.ones((2,), dtype=numpy.float32),
            )


            s = Numeric(
                low=-numpy.ones((2, 2), dtype=numpy.float32),
                high=numpy.ones((2, 2), dtype=numpy.float32),
            )
            assert s[:, 0] == Numeric(
                low=-numpy.ones((2,), dtype=numpy.float32),
                high=numpy.ones((2,), dtype=numpy.float32),
            )
            assert s[0, :] == Numeric(
                low=-numpy.ones((2,), dtype=numpy.float32),
                high=numpy.ones((2,), dtype=numpy.float32),
            )
            assert s[:, :] == s
            assert s[...] == s
        """
        return type(self)(
            low=self.low[key],
            high=self.high[key],
            seed=self.seed,
            dtype=self.dtype,
            contains=self.contains,
        )

    def __eq__(self, other):
        """__eq__


        .. code-block:: python

                s = Numeric(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)))
                assert s == Numeric(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)))
                assert s != Numeric(low=-1.5 * numpy.ones((2, 2)), high=2 * numpy.ones((2, 2)))
                assert s != Numeric(low=-numpy.ones((1,)), high=numpy.ones((1,)))

        :param other: space to compare to
        :type other: Numeric

        """
        if not isinstance(other, type(self)):
            return False
        return (
            self.shape == other.shape
            and (self.low == other.low).all()
            and (self.high == other.high).all()
            and self.dtype == other.dtype
        )

    def __contains__(self, item, mode=None):
        """Check whether ``item`` belongs to the space

        .. code-block:: python

            s = Numeric(
                low=-numpy.ones((2, 2)),
                high=numpy.ones((2, 2)),
            )

            assert [0.0, 0.0, 0.0, 0.0] not in s
            assert [[0.0, 0.0], [0.0, 0.0]] in s
            assert numpy.array([0.0, 0.0, 0.0, 0.0]) not in s
            assert numpy.array([[0.0, 0.0], [0.0, 0.0]]) in s

            assert 1.0 * numpy.ones((2, 2)) in s
            assert -1.0 * numpy.ones((2, 2)) in s

            assert numpy.ones((2, 2), dtype=numpy.int16) in s

        :param item: item
        :type item: numpy.ndarray
        :param mode: see "contains" keyword argument, defaults to None
        :type mode: string, optional
        """

        if mode is None:
            mode = self.contains

        if mode == "numpy":
            try:
                return numpy.all(item >= self.low) and numpy.all(item <= self.high)
            except:
                return False
        else:
            raise NotImplementedError

    def __repr__(self):
        if self.seed is None:
            return f"{type(self).__name__}([{self.low}, {self.high}]) -- {self.dtype}"
        else:
            return f"{type(self).__name__}([{self.low}, {self.high}]) -- {self.dtype} -- seed: {self.seed}"

    def __flat__(self):
        if self.seed is None:
            return f"{type(self).__name__}({self.shape}) -- {self.dtype}"
        else:
            return f"{type(self).__name__}({self.shape}) -- {self.dtype} -- seed: {self.seed}"

    def sample(self):
        """sample

        Generate values by sampling from the interval. If the interval represents integers, sampling is uniform. Otherwise, sampling is Gaussian. You can set the seed to sample, see keyword arguments at init.

        .. code-block:: python

            s = Numeric(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)), seed=123)
            q = Numeric(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)), seed=123)
            r = Numeric(low=-numpy.ones((2, 2)), high=numpy.ones((2, 2)), seed=12)

            _s, _q, _r = s.sample(), q.sample(), r.sample()
            assert _s in s
            assert _q in q
            assert _r in r
            assert (_s == _q).all()
            assert (_s != _r).any()

        """
        if numpy.issubdtype(self.dtype, numpy.integer):

            return self.rng.integers(
                low=self.low, high=self.high, endpoint=True, dtype=self.dtype.type
            )

        else:
            return numpy.nan_to_num(
                (self.high - self.low), nan=1, posinf=1
            ) * self.rng.random(self.shape, dtype=self.dtype.type) + numpy.nan_to_num(
                self.low, neginf=1
            )

    def serialize(self):
        """serialize to JSON"""
        return {
            "space": type(self).__name__,
            "seed": self.seed,
            "low,high": [self.low.tolist(), self.high.tolist()],
            "shape": self.shape,
            "dtype": self.dtype.__class__.__name__,
        }


class CatSet(BaseSpace):
    """Categorical Set

    A categorical set defined explicitly. Use this for data where traditional distance is meaningless i.e. when 1 is not closer to 0 then to 5.
    Performance of this object for large dimensions may be bad, because the whole array is stored in memory.

    .. code-block:: python

        s = space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
        assert s.dtype == numpy.int16
        assert s.N == 3
        assert s.shape == ()

    You can further set the seed of the space (useful when sampling from the space), force the dtype of the space and specify how membership to the space is checked via the keyword arguments. See ``BaseSpace`` for more information.


    :param array: set of values, defaults to None
    :type array: numpy.ndarray, optional
    """

    def __init__(self, array=None, seed=None, dtype=None, contains="numpy"):

        self.array = array

        if dtype is not None:
            if not numpy.issubdtype(dtype, numpy.integer):
                raise ValueError("dtype has to be an integer type")
            self._dtype = numpy.dtype(dtype)
        else:
            self._dtype = None

        super().__init__(seed=seed, contains=contains)
        self.array = array.astype(self.dtype)

    @property
    def N(self):
        """Cardinality of the set"""
        return len(self.array)

    @property
    def dtype(self):
        """numpy.dtype of data"""
        if self._dtype is None:
            if self.array is not None:
                self._dtype = numpy.dtype(self.array.dtype)
                if not numpy.issubdtype(self._dtype, numpy.integer):
                    self._dtype = numpy.dtype(numpy.int64)
        return self._dtype

    @property
    def shape(self):
        """numpy shape of the data that belongs to the set"""
        if self._shape is None:
            self._shape = ()
        return self._shape

    @property
    def low(self):  # Should be removed, doesn't make sense
        return self.array[0]

    @property
    def high(self):  # Should be removed, doesn't make sense
        return self.array[-1]

    def __iter__(self):
        """__iter__"""
        return self

    def __next__(self):
        """__next__"""

        raise StopIteration

    def __getitem__(self, key):
        """__getitem__

        The set can not be separated into different elements, and indexing over the set is only possible in edge cases:

        .. code-block:: python

        s = CatSet(array=numpy.array([1, 2, 3], dtype=numpy.int16))
        s[0] # raises a ``SpaceNotSeparableError``
        assert s[...] == s
        assert s[:] == s
        """
        if key == Ellipsis:
            return self
        if key == slice(None, None, None):
            return self

        from coopihc.base.utils import SpaceNotSeparableError

        raise SpaceNotSeparableError("This space is not separable")

    def __eq__(self, other):
        """__eq__

        .. code-block:: python

            s = CatSet(array=numpy.array([1, 2, 3], dtype=numpy.int16))
            assert s == space(array=numpy.array([1, 2, 3], dtype=numpy.int16))
            assert s != space(array=numpy.array([1, 2, 3, 4], dtype=numpy.int16))


        :param other: other space
        :type other: CatSet

        """
        if not isinstance(other, type(self)):
            return False

        try:
            return (self.array == other.array).all() and self.dtype == other.dtype
        except AttributeError:
            return self.array == other.array and self.dtype == other.dtype

    def __contains__(self, item, mode=None):
        """__contains__

        Checks if item belong to the space. By default, this check is done leniently, according to Numpy __contains__, see kwargs in init for more information.

        .. code-block:: python

            s = CatSet(array=numpy.array([1, 2, 3], dtype=numpy.int16))
            assert 1 in s
            assert [1] in s
            assert [[1]] in s
            assert numpy.array(1) in s
            assert numpy.array([1]) in s
            assert numpy.array([[1]]) in s

            assert numpy.array([2]) in s
            assert numpy.array([3]) in s

            assert numpy.array([1.0]) in s
            assert numpy.array([2]) in s

        """
        if mode is None:
            mode = self.contains

        if mode == "numpy":
            return item in self.array
        else:
            raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.array})"

    def __flat__(self):
        return f"{type(self).__name__}({self.name})"

    def sample(self):
        """sample

        Generate values by sampling uniformly from the set. You can set the seed to the rng, see keyword arguments at init.

        .. code-block:: python

            s = CatSet(array=numpy.arange(1000), seed=123)
            q = CatSet(array=numpy.arange(1000), seed=123)
            r = CatSet(array=numpy.arange(1000), seed=12)
            _s, _q, _r = s.sample(), q.sample(), r.sample()
            assert _s in s
            assert _q in q
            assert _r in r
            assert _s == _q
            assert _s != _r

        """
        return self.rng.choice(self.array)

    def serialize(self):
        return {
            "space": type(self).__name__,
            "seed": self.seed,
            "array": self.array.tolist(),
            "dtype": self.dtype.__class__.__name__,
        }
