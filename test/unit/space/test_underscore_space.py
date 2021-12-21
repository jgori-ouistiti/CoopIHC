from coopihc.space.Space import _Space as Space
from coopihc.space.utils import discrete_space, continuous_space, multidiscrete_space
from coopihc.helpers import flatten
import numpy
import copy


def test_init_discrete():
    # Discrete
    s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
    # prop and attributes
    assert s.dtype == numpy.int16
    assert s.space_type == "discrete"
    assert s.N == 3
    assert s.high == 3
    assert s.low == 1
    assert s.shape == (1,)
    # __ methods
    # __len__
    assert len(s) == 1


def test_contains_soft_discrete():
    s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete", contains="soft")
    assert 1 in s
    assert [1] in s
    assert [[1]] in s
    assert numpy.array(1) in s
    assert numpy.array([1]) in s
    assert numpy.array([[1]]) in s

    assert numpy.array([2]) in s
    assert numpy.array([3]) in s

    assert numpy.array([1.0]) not in s
    assert numpy.array([2], dtype=numpy.float32) not in s


def test_contains_hard_discrete():
    s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete", contains="hard")
    assert 1 not in s
    assert [1] not in s
    assert [[1]] not in s
    assert numpy.array(1) not in s
    assert numpy.array([1]) in s
    assert numpy.array([[1]]) not in s

    assert numpy.array([2]) in s
    assert numpy.array([3]) in s

    assert numpy.array([2.0]) not in s
    assert numpy.array([2], dtype=numpy.float32) not in s


def test_discrete():
    test_init_discrete()
    test_contains_soft_discrete()
    test_contains_hard_discrete()


def test_init_multidiscrete():
    # Discrete
    s = Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )
    # prop and attributes
    assert s.dtype == numpy.int16
    assert s.space_type == "multidiscrete"
    assert s.N == [3, 5]
    assert (s.high == numpy.array([3, 5])).all()
    assert (s.low == numpy.array([1, 1])).all()
    assert s.shape == (2, 1)
    # __ methods
    # __len__
    assert len(s) == 2


def test_contains_soft_multidiscrete():
    s = Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )
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


def test_contains_hard_multidiscrete():
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


def test_multidiscrete():
    test_init_multidiscrete()
    test_contains_soft_multidiscrete()
    test_contains_hard_multidiscrete()


def test_init_continuous():
    # Discrete
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="hard",
    )
    # prop and attributes
    assert s.dtype == numpy.float32
    assert s.space_type == "continuous"
    assert s.N == None
    assert (s.high == numpy.ones((2, 2), dtype=numpy.float32)).all()
    assert (s.low == -numpy.ones((2, 2), dtype=numpy.float32)).all()
    assert s.shape == (2, 2)
    assert len(s) == 4


def test_contains_soft_continuous():
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


def test_contains_hard_continuous():
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


def test_continuous():
    test_init_continuous()
    test_contains_soft_continuous()
    test_contains_hard_continuous()


def test_equal_discrete():
    s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
    assert s == Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
    assert s == numpy.array([1, 2, 3], dtype=numpy.int16)
    assert s != Space(numpy.array([1, 2, 3, 4], dtype=numpy.int16), "discrete")
    assert s == numpy.array([[1, 2, 3]], dtype=numpy.int16)
    assert s == numpy.array([[1, 2, 3]], dtype=numpy.float32)


def test_equal_multidiscrete():
    s = Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )
    assert s == Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )
    assert s == [
        numpy.array([1, 2, 3], dtype=numpy.int16),
        numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
    ]
    assert s != [numpy.array([1, 2, 3], dtype=numpy.int16)]
    assert s != Space(
        [
            numpy.array([1, 2, 3, 5], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )
    assert s != Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )


def test_equal_continuous():
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="soft",
    )
    assert s == Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="soft",
    )
    assert s == [
        -numpy.ones((2, 2), dtype=numpy.float32),
        numpy.ones((2, 2), dtype=numpy.float32),
    ]
    assert s != [
        -1.05 * numpy.ones((2, 2), dtype=numpy.float32),
        numpy.ones((2, 2), dtype=numpy.float32),
    ]
    assert s != Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            1.05 * numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="soft",
    )
    assert s != Space(
        [
            -numpy.ones((3, 2), dtype=numpy.float32),
            numpy.ones((3, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="soft",
    )


def test_equal():
    test_equal_discrete()
    test_equal_multidiscrete()
    test_equal_continuous()


def test_iter_discrete():
    s = Space(numpy.array([1, 2, 3], dtype=numpy.int16), "discrete")
    assert iter(s).__next__() == s


def test_iter_continuous():
    s = Space(
        [
            -numpy.ones((2, 2), dtype=numpy.float32),
            numpy.ones((2, 2), dtype=numpy.float32),
        ],
        "continuous",
        contains="soft",
    )
    for _space in s:
        assert _space == Space(
            [
                -numpy.ones((1, 2), dtype=numpy.float32),
                numpy.ones((1, 2), dtype=numpy.float32),
            ],
            "continuous",
            contains="soft",
        )
        for __space in _space:
            assert __space == Space(
                [
                    -numpy.ones((1, 1), dtype=numpy.float32),
                    numpy.ones((1, 1), dtype=numpy.float32),
                ],
                "continuous",
                contains="soft",
            )


def test_iter_multidiscrete():
    s = Space(
        [
            numpy.array([1, 2, 3], dtype=numpy.int16),
            numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16),
        ],
        "multidiscrete",
        contains="soft",
    )
    iterator = iter(s)
    assert iterator.__next__() == Space(
        numpy.array([1, 2, 3], dtype=numpy.int16), "discrete", contains="soft"
    )
    assert s.n == 1
    assert iterator.__next__() == Space(
        numpy.array([1, 2, 3, 4, 5], dtype=numpy.int16), "discrete", contains="soft"
    )


def test_iter():
    test_iter_discrete()
    test_iter_continuous()
    test_iter_multidiscrete()


if __name__ == "__main__":
    test_discrete()
    test_multidiscrete()
    test_continuous()
    test_equal()
    test_iter()


# def test_sample():
#     # ================== Discrete
#     number_of_targets = 3
#     gridsize = [5, 5]
#     s = Space(
#         flatten(
#             [
#                 [
#                     numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
#                     numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
#                 ]
#                 for j in range(number_of_targets)
#             ]
#         ),
#         seed=123,
#     )

#     # -------- Check if samples are valid
#     for i in range(100):
#         assert s.sample() in s
#     # --------- Check two samples are different
#     assert (s.sample() != s.sample()).any()
#     # --------- Check that seeding works
#     ss = copy.deepcopy(s)
#     assert (ss.sample() == s.sample()).all()
#     sss = Space(
#         flatten(
#             [
#                 [
#                     numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
#                     numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
#                 ]
#                 for j in range(number_of_targets)
#             ]
#         ),
#         seed=123,
#     )
#     ssss = Space(
#         flatten(
#             [
#                 [
#                     numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
#                     numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
#                 ]
#                 for j in range(number_of_targets)
#             ]
#         ),
#         seed=123,
#     )

#     sfive = Space(
#         flatten(
#             [
#                 [
#                     numpy.array([i for i in range(gridsize[0])], dtype=numpy.int16),
#                     numpy.array([i for i in range(gridsize[1])], dtype=numpy.int16),
#                 ]
#                 for j in range(number_of_targets)
#             ]
#         ),
#         seed=13,
#     )

#     a1 = sss.sample()
#     a2 = ssss.sample()
#     a3 = sfive.sample()
#     assert (a1 == a2).all()
#     assert (a1 != a3).any()
#     # =============== Continuous
#     s = Space(
#         [
#             -numpy.ones((2, 2), dtype=numpy.float32),
#             numpy.ones((2, 2), dtype=numpy.float32),
#         ]
#     )
#     # -------- Check if samples are valid
#     for i in range(100):
#         assert s.sample() in s
#     # --------- Check two samples are different
#     assert (s.sample() != s.sample()).any()
#     # --------- Check that seeding works
#     ss = copy.deepcopy(s)
#     assert (ss.sample() == s.sample()).all()


# def test_eq():
#     s = Space(
#         [
#             -numpy.ones((2, 2), dtype=numpy.float32),
#             numpy.ones((2, 2), dtype=numpy.float32),
#         ]
#     )
#     v = Space(
#         [
#             -numpy.ones((2, 2), dtype=numpy.float32),
#             numpy.ones((2, 2), dtype=numpy.float32),
#         ]
#     )
#     w = Space(
#         [
#             -numpy.ones((2, 2), dtype=numpy.float64),
#             numpy.ones((2, 2), dtype=numpy.float64),
#         ]
#     )
#     assert s == v
#     assert s != w


# def test_getitem():
#     # discrete
#     s = Space(
#         [
#             numpy.array([1, 2, 3], dtype=numpy.int16),
#         ]
#     )
#     assert s[0] == s
#     # continuous
#     print("\n================\n")
#     s = Space(
#         [
#             -numpy.ones((2, 2), dtype=numpy.float32),
#             numpy.ones((2, 2), dtype=numpy.float32),
#         ]
#     )
#     assert s[0] == s
#     # multidiscrete
#     s = Space(
#         [
#             numpy.array([3, 4, 5], dtype=numpy.int16),
#             numpy.array([1, 2, 3], dtype=numpy.int16),
#         ]
#     )
#     assert s[0] == Space(
#         [
#             numpy.array([3, 4, 5], dtype=numpy.int16),
#         ]
#     )
#     assert s[1] == Space(
#         [
#             numpy.array([1, 2, 3], dtype=numpy.int16),
#         ]
#     )


# def test_shortcuts():
#     space = discrete_space([1, 2, 3])
#     space = continuous_space(-numpy.eye(2, 2), numpy.eye(2, 2))
#     space = multidiscrete_space([[1, 2, 3], [4, 5, 6]])
