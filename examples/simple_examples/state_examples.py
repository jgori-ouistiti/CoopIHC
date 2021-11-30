if __name__ == "__main__":

    import numpy
    from coopihc.space.Space import Space
    from coopihc.space.State import State
    from coopihc.space.StateElement import StateElement

    numpy.set_printoptions(precision=3, suppress=True)

    # [start-state-example]
    # Continuous substate. Provide Space([low, high]). Value is optional
    x = StateElement(
        values=None,
        spaces=Space(
            [
                numpy.array([-1.0]).reshape(1, 1),
                numpy.array([1.0]).reshape(1, 1),
            ]
        ),
    )

    # Discrete substate. Provide Space([range]). Value is optional
    y = StateElement(values=2, spaces=Space([numpy.array([1, 2, 3], type=numpy.int)]))

    # Define a State, composed of two substates previously defined
    s1 = State(substate_x=x, substate_y=y)

    # Define a super-State that is composed of the State previously defined
    S = State()
    S["substate1"] = s1
    # [end-state-example]

    # [start-state-init]
    x = StateElement(
        values=1,
        spaces=Space(
            [
                numpy.array([-1.0]).reshape(1, 1),
                numpy.array([1.0]).reshape(1, 1),
            ]
        ),
    )

    y = StateElement(values=2, spaces=Space([numpy.array([1, 2, 3], type=numpy.int)]))

    z = StateElement(
        values=5,
        spaces=Space([numpy.array([i for i in range(10)], dtype=numpy.int)]),
    )

    s1 = State(substate_x=x, substate_y=y, substate_z=z)

    w = StateElement(
        values=numpy.zeros((3, 3)),
        spaces=Space([-3.5 * numpy.ones((3, 3)), 6 * numpy.ones((3, 3))]),
    )
    s1["substate_w"] = w

    xx = StateElement(
        values=numpy.ones((2, 2)),
        spaces=Space([-0.5 * numpy.ones((2, 2)), 0.5 * numpy.ones((2, 2))]),
        clipping_mode="clip",
    )

    yy = StateElement(
        values=None,
        spaces=Space(
            [numpy.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6], dtype=numpy.int16)]
        ),
    )

    s2 = State(**{"substate_xx": xx, "substate_yy": yy})

    S = State()
    S["substate1"] = s1
    S["substate2"] = s2
    # [end-state-init]

    # [start-state-reset]
    print(S.reset())
    # [end-state-reset]

    # [start-state-filter]
    from collections import OrderedDict

    ordereddict = OrderedDict(
        {"substate1": OrderedDict({"substate_x": 0, "substate_w": 0})}
    )

    ns1 = S.filter("values", filterdict=ordereddict)
    ns2 = S.filter("spaces", filterdict=ordereddict)
    ns5 = S.filter("values")
    ns6 = S.filter("spaces")

    # [end-state-filter]

    # [start-state-serialize]
    print(S.serialize())
    # [end-state-serialize]
