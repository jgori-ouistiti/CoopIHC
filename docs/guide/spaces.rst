.. spaces:


Spaces
===========
Spaces are used by ``StateElement``, to specify in which domain a substate can live. Spaces essentially build on Numpy arrays.


Spaces can be continuous. In that case, you have to specify lower and upper bounds:

.. code-block:: python

    continuous_space = Space( [
        -numpy.ones((2,2), dtype = numpy.float32),
        numpy.ones((2,2), numpy.float32)  ]
            )

Spaces can also be discrete; in that case you have to specify the range of possible values:

.. code-block:: python

    discrete_space = Space(
        [numpy.array([1,2,3], dtype = numpy.int16)]
            )
.. note::

    To avoid ambiguities between a discrete space with 2 items and a continuous space, ``space`` looks at the data type. If you want to use continuous spaces, specify **either floating point values or specify a floating point dtype** in the Numpy array.

You will rarely need to deal directly with spaces except when defining a new substate, but below are some useful things to know about spaces.
You can check if a value is inside a space (once again, notice how the dtype is important).

.. code-block:: python

    space = Space(  [numpy.array([1,2,3], dtype = numpy.int16)] )
    x = numpy.array([2], dtype = numpy.int16)
    y = numpy.array([2], dtype = numpy.float32)
    yy = numpy.array([2])
    z = numpy.array([5])
    assert(x in space)
    assert(y not in space)
    assert(yy in space)
    assert(z not in space)

    space = Space(  [-numpy.ones((2,2), dtype = numpy.float32), numpy.ones((2,2), numpy.float32)]  )
    x = numpy.array([[1,1],[1,1]], dtype = numpy.int16)
    y = numpy.array([[1,1],[1,1]], dtype = numpy.float32)
    yy = numpy.array([[1,1],[1,1]])
    yyy = numpy.array([[1.0,1.0],[1.0,1.0]])
    z = numpy.array([[5,1],[1,1]], dtype = numpy.float32)
    assert(x in space)
    assert(y in space)
    assert(yy in space)
    assert(yyy in space)
    assert(z not in space)

You can also (uniformly) sample from a space

.. code-block:: python

    space = Space(  [numpy.array([[-2,-2],[-1,-1]], dtype = numpy.float32), numpy.ones((2,2), numpy.float32)]  )
    space.sample()

More detailed information is found :doc:`here<../modules/core/space>`
