.. stateelements:


States and StateElements
==========================
States and StateElements are used in virtually any component in *CoopIHC*

* States are containers that contain either other states, or StateElements.
* StateElements are containers that contain

    * values, i.e. the values of the component stored in that StateElement.
    * spaces, i.e. the domains in which these values live.


StateElement
----------------
Defining a StateElement is straightforward, and is achieved by specifying the values/spaces couple to the StateElement constructor.

.. code-block:: python

    continuous_state_element = StateElement(
        values = 0,
        spaces =
            core.space.Space([
            numpy.array([-1], dtype = numpy.float32), numpy.array([1], dtype = numpy.float32)
            ]),
        clipping_mode = 'warning',
        typing_priority = 'space'
    )

    discrete_state_element = StateElement(
        values = 5,
        spaces = core.space.Space(
            [numpy.array([-5,-4,-3,-2,-1,0,1,2,3,4,5], dtype = numpy.int16)]
        )
    )

When instantiating, StateElement verifies that the values and the spaces are valid:

* It casts the data to a common type (see the ``'typing_priority'`` keyword)
* It does something when the values are not contained inside the space, depending on the value of the ``'clipping_mode'`` keyword argument:

    + if ``clipping_mode = 'warning'``, it will issue a warning
    + if ``clipping_mode = 'clip'``, it will automatically clip the values so as to be contained inside the space
    + if ``clipping_mode = 'error'``, it will raise a ``StateNotContainedError``


Accessing and setting data of a StateElement is achieved by itemizing. In the example below, notice how the data is cast to float, and how the warning is triggered for data outside the space.

.. code-block:: python

    >>> a['values']
    [array([[0.]], dtype=float32)]
    >>> a['values'] = numpy.array([1])
    >>> a['values']
    [array([[1.]], dtype=float32)]
    >>> a['values'] = numpy.array([10])
    Warning: Instantiated Value [[10.]](<class 'numpy.ndarray'>) is not contained in corresponding space Space[Continuous((1, 1)), float32] (low = [-1.], high = [1.])
    >>> a['values']
    [array([[10.]], dtype=float32)]


Many useful operations have been defined on StateElements:

* You can iterate over them
* You can perform many arithmetic operations, including matrix multiplication
* You can perform the cartesian product of its spaces,
* You can randomize its values
* You can cast values of a StateElement in some space to a different space, including to and from continuous to and from discrete spaces.

Find more information :doc:`here<../modules/core/stateelements>`.



State
-------------
States derive from dictionaries and their syntax are very much alike. Below, we show how to define a super state which contain states which themselves contain stateelements, showing various possible syntaxes.


.. code-block:: python


    x = StateElement(values = 1,spaces = Space([  numpy.array([-1.0]).reshape(1,1),numpy.array([1.0]).reshape(1,1)  ]))

    y = StateElement(values = 2,spaces = Space( [numpy.array([1,2,3], dtype = numpy.int)]))

    z = StateElement(values = 5,spaces = Space( [numpy.array([i for i in range(10)], dtype = numpy.int)]))


    s1 = State(
        substate_x = x,
        substate_y = y,
        substate_z = z
    )

    w = StateElement(values = numpy.zeros((3,3)),
        spaces = Space([-3.5*numpy.ones((3,3)),6*numpy.ones((3,3))])
    )
    s1['substate_w'] = w

    xx = StateElement(values = numpy.ones((2,2)),spaces = Space([-0.5*numpy.ones((2,2)),0.5*numpy.ones((2,2))]),clipping_mode = 'clip'
    )

    yy = StateElement(values = None,spaces = Space( [numpy.array([-3,-2,-1,0,1,2,3,4,5,6])]))


    s2 = State(
        **{
        'substate_xx': xx,
        'substate_yy': yy
        }
    )

    S = State()
    S['substate1'] = s1
    S['substate2'] = s2


States can be reset (in which case new values for statelements are randomly sampled) and filtered, see below:

.. code-block:: python

    >>> S.reset()
    >>> print(S)
      Index  Label                    Value                     Space
    -------  -----------------------  ------------------------  ----------------------------------
          0  substate1|substate_x|0   [[0.276]]                 Space[Continuous((1, 1)), float64]
          1  substate1|substate_y|0   [[2]]                     Space[Discrete(3), int64]
          2  substate1|substate_z|0   [[5]]                     Space[Discrete(10), int64]
          3  substate1|substate_w|0   [[-0.136 -2.548 -0.808]   Space[Continuous((3, 3)), float64]
                                       [-2.127 -1.433 -2.978]
                                       [-1.607  1.93   0.555]]
          4  substate2|substate_xx|0  [[ 0.36   0.054]          Space[Continuous((2, 2)), float64]
                                       [ 0.487 -0.162]]
          5  substate2|substate_yy|0  [[1]]                     Space[Discrete(10), int64]


.. code-block:: python

    from collections import OrderedDict
    ordereddict = OrderedDict({ 'substate1' : OrderedDict({'substate_x': 0, 'substate_w': 0})})

    ns1 = S.filter('values', filterdict = ordereddict)
    ns2 = S.filter('spaces', filterdict = ordereddict)
    ns5 = S.filter('values')
    ns6 = S.filter('spaces')


Find more information :doc:`here<../modules/core/states>`.



States are used to represent the game state in *CoopIHC*, as illustrated below:

.. tikz:: How states and stateelements are used to represent the game state in *CoopIHC*
    :include: tikz/states.tikz
    :xscale: 100
    :align: center
