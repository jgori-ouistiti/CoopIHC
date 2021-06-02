.. states:

States and StateElements
==========================

States and StateElements are basic components used throughout bundles.

* States are containers that contain either other states, or StateElements.
* StateElements are containers that contain

    * values, i.e. the values of the component stored in that StateElement
    * spaces, i.e. the spaces in which these values are contained
    * possible values, which are the true values which are meaningful to a human.

The game state in a bundle is represented below. It is composed by aggregating states of all components of the bundles, which themselves hold several instances of StateElement.

.. tikz:: State and StateElement in the game state
    :include: tikz/states.tikz
    :xscale: 100
    :align: center


StateElement
----------------

StateElement is a type of container which hold values, spaces, possible values, as described above and provide some facilitating methods.

The usual way to initialize a StateElement is to call it with the keyword arguments values, spaces and possible_values provided:

.. code-block:: python

    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]]
                        )

The following is expected, from a StateElement of dimension N

* spaces should be a length N list, where each element indicates the space in which the nth value is contained,
* values should be a length N list, where each element indicates the nth value,
* possible_values should be a length N list, where each element is a list containing the mapping from each value to each possible_values element


StateElement performs some checks and conversions on the data that it is being input, so that it is robust to syntax differences. Nones can also be used if a field is not applicable (e.g. values are not known yet, or the values are already in the true human format).

The examples below will all work, even though they don't adhere to the principles above.

.. code-block:: python

    # Not in list format
    x = StateElement(   values = numpy.array([-.5]).reshape(1,),
        spaces = gym.spaces.Box(-1,1, shape = (1,)),
        possible_values = None)

    # First value should be of type array
    x = StateElement(   values = [1,2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]]
                        )
    # Same + None used for possible_values
    y = StateElement(   values = [1,2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = None)

    # values as nested lists
    x = StateElement(   values = [[1],[2], [3]],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])

    # Various Nones
    x = StateElement(values = None, spaces = [gym.spaces.Discrete(2)], possible_values  = None)
    y = StateElement()
    z = StateElement(
                values = [None, None],
                spaces = [gym.spaces.Box(low = -numpy.inf, high = numpy.inf, shape = (1,)) for i in range(2)],
                possible_values = None
                 )


The different fields can then be accessed as attributes, or via indexing.

.. code-block:: python

    x = StateElement(   values = [[1],[2], [3]],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    >>> x['values']
    [array([1]), 2, 3]
    >>> x.values
    [array([1]), 2, 3]
    >>> x['values'] = [1, 1, 1]
    >>> x['values']
    [array([1]), 1, 1]


Useful methods
^^^^^^^^^^^^^^^^^

* :ref:`reset-label`
* :ref:`get-human-values-label`
* :ref:`iter-label`
* :ref:`cartesian-product-label`




.. _reset-label:

reset()
"""""""""""
States can be reset, in which case values are filled by sampling from the spaces.

.. code-block:: python

    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]] )

    >>> x

    value:	[array([1]), 2, 3]
    spaces:	[Box(1,), Discrete(3), Discrete(6)]
    possible values:	[[None], [None], [-6, -5, -4, -3, -2, -1]]

    x.reset()
    >>> x

    value:	[array([-0.3772682], dtype=float32), 2, 4]
    spaces:	[Box(1,), Discrete(3), Discrete(6)]
    possible values:	[[None], [None], [-6, -5, -4, -3, -2, -1]]

States can also be forced to be reset to particular values, by providing a dictionary:

.. code-block:: python

    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    reset_dic = {'values': [-1/2,-0,0]}
    x.reset(dic = reset_dic)
    >>> x

    value:	[array([-0.5]), 0, 0]
    spaces:	[Box(1,), Discrete(3), Discrete(6)]
    possible values:	[[None], [None], [-6, -5, -4, -3, -2, -1]]

.. _get-human-values-label:

get_human_values()
"""""""""""""""""""

In various places, the modeler is expected to provide models (e.g. a task model, a user model), in which case it is easier to allow the modeler to specify values in a format that he likes. This is why the possible_values field exists. To get access the values converted in the human readable format, one can call the ``get_human_values()`` method. There is also a special way of accessing this method via indexing.

.. code-block:: python

    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])

    # via the method
    >>> x.get_human_values()
    [array([1]), 2, -3]
    # via indexing
    >>> x['human_values']
    [array([1]), 2, -3]


.. _iter-label:

iter()
""""""""""""
StateElements can be iterated  upon:

.. code-block:: python

    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])
    >>> for _x in x:
    ...         print(_x)
    ...

    value:	[array([1])]
    spaces:	[Box(1,)]
    possible values:	[None]


    value:	[2]
    spaces:	[Discrete(3)]
    possible values:	[None]


    value:	[3]
    spaces:	[Discrete(6)]
    possible values:	[[-6, -5, -4, -3, -2, -1]]

.. _cartesian-product-label:

cartesian_product()
"""""""""""""""""""""""
One can also get the cartesian product of StateElement, (where values in continuous domains remain constant):

.. code-block:: python

    x = StateElement(   values = [numpy.array([1]).reshape(1,),2,3],
                        spaces = [gym.spaces.Box(-1,1, shape = (1,)), gym.spaces.Discrete(3), gym.spaces.Discrete(6)],
                        possible_values = [[None], [None], [-6,-5,-4,-3,-2,-1]])

    >>> for _x in x.cartesian_product():
    ...         print(_x)
    ...

    value:	[array([1]), 0, 0]
    spaces:	[Box(1,), Discrete(3), Discrete(6)]
    possible values:	[[None], [None], [-6, -5, -4, -3, -2, -1]]


    value:	[array([1]), 0, 1]
    spaces:	[Box(1,), Discrete(3), Discrete(6)]
    possible values:	[[None], [None], [-6, -5, -4, -3, -2, -1]]


    [...]

    value:	[array([1]), 2, 4]
    spaces:	[Box(1,), Discrete(3), Discrete(6)]
    possible values:	[[None], [None], [-6, -5, -4, -3, -2, -1]]


    value:	[array([1]), 2, 5]
    spaces:	[Box(1,), Discrete(3), Discrete(6)]
    possible values:	[[None], [None], [-6, -5, -4, -3, -2, -1]]


State
-------------

State is a subclass of ``OrderedDict``, which adds a reset method and modifies the default __repr__ and __deepcopy__ methods.
