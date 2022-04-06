import numpy
from coopihc.base.elements import array_element, discrete_array_element, cat_element


def test_init():
    array_element(init=1)
    discrete_array_element(N=5, init=1)
    cat_element(
        5,
        init=0,
    )
    discrete_array_element(init=-4, low=-4, high=4)
    array_element(shape=(1, 1))
    discrete_array_element(low=0, high=31, shape=(8,))


if __name__ == "__main__":
    test_init()
