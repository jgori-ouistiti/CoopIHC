.. space:

Space, StateElement, State
===========================

Space
---------------
``Space`` s are used by :ref:`StateElement` s, to specify in which domain a substate can live. Space ranges are defined by NumPy arrays. The complete interface is explained in the :py:class:`API Reference<coopihc.space.Space.Space>`.

Defining spaces
^^^^^^^^^^^^^^^^^

Spaces can be either continuous, discrete or multidiscrete. For continuous spaces, you have to specify lower and upper bounds, for discrete and multidiscrete spaces you have to specify the array of possible values. Some shortcuts exist to make defining Spaces less cumbersome.

.. literalinclude:: ../../coopihc/examples/simple_examples/space_examples.py
   :language: python
   :linenos:
   :start-after: [start-space-def]
   :end-before: [end-space-def]

A global shortcut function called ``autospace`` exists that lets you define any space in a less rigorous way, at the cost of some extra computation:

.. literalinclude:: ../../coopihc/examples/simple_examples/space_examples.py
   :language: python
   :linenos:
   :start-after: [start-space-autospace]
   :end-before: [end-space-autospace]



.. note::

    If Numpy's ``dtype`` is not given, ``Space`` will assume you are using ``numpy.int16`` or ``numpy.float32`` for respectively (multi)discrete and continuous spaces. However, this is just a convention, and it is perfectly fine for the most part to have a continuous space with e.g. ``numpy.uint8`` data types.

Mechanisms
^^^^^^^^^^^^

``Space``\s come with several mechanisms: 

1. check whether a value is contained in a space,
2. sample a value from a space,
3. iterate over a space,
4. Perform the cartesian product of spaces.


Values ``in`` space
************************
You can check if a value belongs to a space with Python's ``in`` operator:

.. literalinclude:: ../../coopihc/examples/simple_examples/space_examples.py
   :language: python
   :linenos:
   :start-after: [start-space-contains]
   :end-before: [end-space-contains]

Notice that whether a value is in the space or not also depends on its data type and the data type of the space. In particular, a ``numpy.int16`` can always be represented as a ``numpy.float32``, and so a ``numpy.int16`` value can be contained in a space with data type ``numpy.float32``, but the opposite is not true.

.. note::

   ``Space`` has a ``contains`` keyword argument, which can be set to 'hard'. In the default 'soft' case, Space will try to broadcast or viewcast the input to match with the space arrays, but not in 'hard' mode. Examples can be found :py:class:`here<coopihc.space.Space.Space>`.

Sampling from spaces
**************************
You can sample values from spaces using the ``sample()`` method:

.. literalinclude:: ../../coopihc/examples/simple_examples/space_examples.py
   :language: python
   :linenos:
   :start-after: [start-space-sample]
   :end-before: [end-space-sample]

You can provide a given seed for each ``Space`` via its ``seed`` keyword argument, for repeatable random experiments.



Iterating over spaces
*************************
You can iterate over spaces, potentially several times:

.. literalinclude:: ../../coopihc/examples/simple_examples/space_examples.py
   :language: python
   :linenos:
   :start-after: [start-space-iter]
   :end-before: [end-space-iter]


.. note::

   Iterating over a discrete space returns itself. Iterating over a continuous space is comparable to iterating over Numpy arrays: iteration is over rows, then columns. To iterate element-wise over a continuous space, you can use regular indexing. In short, remember that ``continuous_space[0] != next(iter(continuous_space))``.


Cartesian Product
********************
You can compute the cartesian product of several spaces. For continuous spaces, a ``None`` value is used.

.. literalinclude:: ../../coopihc/examples/simple_examples/space_examples.py
   :language: python
   :linenos:
   :start-after: [start-space-cp]
   :end-before: [end-space-cp]


StateElement
-----------------

StateElements are the lowest level substates in *CoopIHC*. They are containers that hold

    * the **values** of the component stored in that ``StateElement``.
    * the **spaces** in which these values live, defined as ``Space`` objects.


You define a StateElement by specifying the values/spaces couple to the StateElement constructor.

.. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
   :language: python
   :linenos:
   :start-after: [start-stateelement-init]
   :end-before: [end-stateelement-init]

When instantiating, StateElement verifies that the values and the spaces are valid:

    * It casts the data to a common type (see the ``'typing_priority'`` keyword)
    * It does something when the values are not contained inside the space, depending on the value of the ``'clipping_mode'`` keyword argument:

        + if ``clipping_mode = 'warning'``, it will issue a warning
        + if ``clipping_mode = 'clip'``, it will automatically clip the values so as to be contained inside the space
        + if ``clipping_mode = 'error'``, it will raise a ``StateNotContainedError``


You can set and get data with either itemization or the dot syntax. In all cases, the input data is processed and corrected if it does not meet the expected syntax or is not contained in the space.


Many useful operations have been defined on StateElements:

+ **You can randomize its values**
    .. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
       :language: python
       :linenos:
       :start-after: [start-stateelement-reset]
       :end-before: [end-stateelement-reset]


+ **You can iterate over them**

    .. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
       :language: python
       :linenos:
       :start-after: [start-stateelement-iter]
       :end-before: [end-stateelement-iter]


+ **You can perform many arithmetic operations, including matrix multiplication**

    .. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
       :language: python
       :linenos:
       :start-after: [start-stateelement-arithmetic]
       :end-before: [end-stateelement-arithmetic]

+ **You can perform logical comparisons**

    .. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
       :language: python
       :linenos:
       :start-after: [start-stateelement-comp]
       :end-before: [end-stateelement-comp]

+ **You can perform the cartesian product of its spaces**

    .. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
       :language: python
       :linenos:
       :start-after: [start-stateelement-cp]
       :end-before: [end-stateelement-cp]

+ **You can cast values of a StateElement, to and from continuous to and from discrete spaces.**

    .. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
       :language: python
       :linenos:
       :start-after: [start-stateelement-cast]
       :end-before: [end-stateelement-cast]





State
------------
States are the higher level container used in *CoopIHC*, that can contain either StateElements or other States. They derive from collections.OrderedDict

    + as a result their syntax are identical
    + the existing methods and operators defined for OrderedDict also work on states (e.g. iteration, keys(), items() etc. possible)

Defining a State is straightforward

.. literalinclude:: ../../coopihc/examples/simple_examples/state_examples.py
   :language: python
   :linenos:
   :start-after: [start-state-init]
   :end-before: [end-state-init]

States can be initialized to a random values (or forced, like StateElements)

.. literalinclude:: ../../coopihc/examples/simple_examples/state_examples.py
   :language: python
   :linenos:
   :start-after: [start-state-reset]
   :end-before: [end-state-reset]

States can also be filtered by providing an OrderedDict of items that you would like to retain

.. literalinclude:: ../../coopihc/examples/simple_examples/state_examples.py
   :language: python
   :linenos:
   :start-after: [start-state-filter]
   :end-before: [end-state-filter]

States can also be serialized to a a dictionnary

.. literalinclude:: ../../coopihc/examples/simple_examples/state_examples.py
   :language: python
   :linenos:
   :start-after: [start-state-serialize]
   :end-before: [end-state-serialize]

