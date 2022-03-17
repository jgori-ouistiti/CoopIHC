import numpy
from coopihc.base.elements import array_element, discrete_array_element, cat_element


def test_init():
    array_element(init=1)
    discrete_array_element(5, init=1)
    cat_element(
        5,
        init=0,
    )


if __name__ == "__main__":
    test_init()
