.. space:

Space, StateElement, State
===========================

Space
---------------
Spaces are used by ``StateElement``, to specify in which domain a substate can live. Spaces essentially build on NumPy arrays.


Spaces can be continuous or discrete. In each case, you have to specify the range of values (lower and upper bounds and possible values for resp. continuous and discrete spaces:

.. literalinclude:: ../../core/space.py
   :language: python
   :linenos:
   :start-after: [start-space-def]
   :end-before: [end-space-def]



Spaces with subspaces can also be defined, as well as spaces with single values:

.. literalinclude:: ../../core/space.py
   :language: python
   :linenos:
   :start-after: [start-space-complex-def]
   :end-before: [end-space-complex-def]

.. note::

    ``Space`` looks at the data type to determine whether a space is continuous or discrete. For example, If you want to use continuous spaces, specify
        * either floating point values
        * or specify a floating point dtype for the Numpy array.



You can check if a value belongs to a space with Python's ``in`` operator:

.. literalinclude:: ../../core/space.py
   :language: python
   :linenos:
   :start-after: [start-space-contains]
   :end-before: [end-space-contains]


You can sample values from spaces using the ``sample()`` method:

.. literalinclude:: ../../core/space.py
   :language: python
   :linenos:
   :start-after: [start-space-sample]
   :end-before: [end-space-sample]



You can iterate over spaces, potentially several times:

.. literalinclude:: ../../core/space.py
   :language: python
   :linenos:
   :start-after: [start-space-iter]
   :end-before: [end-space-iter]


.. autoclass:: core.space.Space
    :members:


StateElement
-----------------

StateElements are the lowest level substates in *CoopIHC*. They are containers that hold

    * the **values** of the component stored in that ``StateElement``.
    * the **spaces** in which these values live, defined as ``Space`` objects.


You define a StateElement by specifying the values/spaces couple to the StateElement constructor.

.. literalinclude:: ../../core/space.py
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
    .. literalinclude:: ../../core/space.py
       :language: python
       :linenos:
       :start-after: [start-stateelement-reset]
       :end-before: [end-stateelement-reset]


+ **You can iterate over them**

    .. literalinclude:: ../../core/space.py
       :language: python
       :linenos:
       :start-after: [start-stateelement-iter]
       :end-before: [end-stateelement-iter]


+ **You can perform many arithmetic operations, including matrix multiplication**

    .. literalinclude:: ../../core/space.py
       :language: python
       :linenos:
       :start-after: [start-stateelement-arithmetic]
       :end-before: [end-stateelement-arithmetic]

+ **You can perform logical comparisons**

    .. literalinclude:: ../../core/space.py
       :language: python
       :linenos:
       :start-after: [start-stateelement-comp]
       :end-before: [end-stateelement-comp]

+ **You can perform the cartesian product of its spaces**

    .. literalinclude:: ../../core/space.py
       :language: python
       :linenos:
       :start-after: [start-stateelement-cp]
       :end-before: [end-stateelement-cp]

+ **You can cast values of a StateElement, to and from continuous to and from discrete spaces.**

    .. literalinclude:: ../../core/space.py
       :language: python
       :linenos:
       :start-after: [start-stateelement-cast]
       :end-before: [end-stateelement-cast]


.. autoclass:: core.space.StateElement
    :members:


State
------------
States are the higher level container used in *CoopIHC*, that can contain either StateElements or other States. They derive from collections.OrderedDict

    + as a result their syntax are identical
    + the existing methods and operators defined for OrderedDict also work on states (e.g. iteration, keys(), items() etc. possible)

Defining a State is straightforward

.. literalinclude:: ../../core/space.py
   :language: python
   :linenos:
   :start-after: [start-state-init]
   :end-before: [end-state-init]

States can be initialized to a random values (or forced, like StateElements)

.. literalinclude:: ../../core/space.py
   :language: python
   :linenos:
   :start-after: [start-state-reset]
   :end-before: [end-state-reset]

States can also be filtered by providing an OrderedDict of items that you would like to retain

.. literalinclude:: ../../core/space.py
   :language: python
   :linenos:
   :start-after: [start-state-filter]
   :end-before: [end-state-filter]

States can also be serialized to a a dictionnary

.. literalinclude:: ../../core/space.py
   :language: python
   :linenos:
   :start-after: [start-state-serialize]
   :end-before: [end-state-serialize]


.. autoclass:: core.space.State
    :members:
