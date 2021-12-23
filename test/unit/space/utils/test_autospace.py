import numpy
from coopihc.space.Space import Space
from coopihc.space.utils import autospace


def test_discrete():
    assert autospace([1, 2, 3]) == Space(numpy.array([1, 2, 3]), "discrete")
    assert autospace([[1, 2, 3]]) == Space(numpy.array([1, 2, 3]), "discrete")
    assert autospace(numpy.array([1, 2, 3])) == Space(
        numpy.array([1, 2, 3]), "discrete"
    )
    assert autospace(numpy.array([[1, 2, 3]])) == Space(
        numpy.array([1, 2, 3]), "discrete"
    )
    assert autospace([numpy.array([1, 2, 3])]) == Space(
        numpy.array([1, 2, 3]), "discrete"
    )
    assert autospace([numpy.array([[1, 2, 3]])]) == Space(
        numpy.array([1, 2, 3]), "discrete"
    )


def test_multidiscrete():
    assert autospace(numpy.array([[1, 2, 3], [4, 5, 6]])) == Space(
        [numpy.array([1, 2, 3]), numpy.array([4, 5, 6])], "multidiscrete"
    )
    assert autospace([1, 2, 3], [1, 2, 3, 4, 5]) == Space(
        [numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])], "multidiscrete"
    )

    assert autospace([numpy.array([1, 2, 3])], [numpy.array([1, 2, 3, 4, 5])]) == Space(
        [numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])], "multidiscrete"
    )
    assert autospace(numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])) == Space(
        [numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])], "multidiscrete"
    )
    assert autospace([numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])]) == Space(
        [numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])], "multidiscrete"
    )
    assert autospace(
        [numpy.array([[1, 2, 3]]), numpy.array([[1, 2, 3, 4, 5]])]
    ) == Space([numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5])], "multidiscrete")

    assert autospace([1, 2, 3], [1, 2, 3, 4, 5], [1, 8]) == Space(
        [numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5]), numpy.array([1, 8])],
        "multidiscrete",
    )
    assert autospace(
        [
            numpy.array([[1, 2, 3]]),
            numpy.array([[1, 2, 3, 4, 5]]),
            numpy.array([[1, 8]]),
        ]
    ) == Space(
        [numpy.array([1, 2, 3]), numpy.array([1, 2, 3, 4, 5]), numpy.array([1, 8])],
        "multidiscrete",
    )


def test_continuous():
    assert autospace(-numpy.array([[1, 1], [1, 1]]), numpy.array([[1, 1], [1, 1]]))
    assert autospace([-numpy.array([[1, 1], [1, 1]]), numpy.array([[1, 1], [1, 1]])])
    assert autospace([[-1, -1], [-1, -1]], [[1, 1], [1, 1]])
    assert autospace([[[-1, -1], [-1, -1]], [[1, 1], [1, 1]]])


def test_base_init():
    test_discrete()
    test_multidiscrete()
    test_continuous()


if __name__ == "__main__":
    test_base_init()
