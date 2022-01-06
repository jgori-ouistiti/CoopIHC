.. space:

StateElement
-----------------

A ``StateElement`` is a a combination of a value and a space. Under the hood, ``StateElement`` subclasses ``numpy.ndarray``; essentially, it adds a layer that checks whether the values are contained in the space (and what to do if not). As a result, many NumPy methods will work. A few simple examples are provided below.

.. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
   :language: python
   :linenos:
   :start-after: [start-stateelement-init]
   :end-before: [end-stateelement-init]

``StateElement`` has an ``out_of_bounds_mode`` keyword argument (defaults to 'warning') that specifies what to do when a value is not contained in the space:

* "error" --> raises a ``StateNotContainedError``
* "warning" --> warns with a ``StateNotContainedWarning``, but accepts the input
* "clip" --> clips the data to force it to belong to the space
* "silent" --> Values not in the space are accepted silently (behavior should become equivalent to numpy.ndarray). 
* "raw" --> No data transformation is applied. This is faster than the other options, because the preprocessing of input data is short-circuited. However, this provides no tolerance for misspecified input.

.. note:: 
   
   Broad/viewcasting and type casting are applied if necessary to the first four cases if the space has "contains" set to "soft", but never with "raw". 

Using NumPy functions
------------------------
There are several ways to use Numpy functions directly on StateElements. The easiest case if when you are okay with losing the information pertaining to space. In that case, you can just work with the array directly by casting the stateelement to an array:

.. code-block:: python
   :linenos:

   cont_space = autospace(
    [[-1, -1], [-1, -1]], [[1, 1], [1, 1]], dtype=numpy.float64
   )  
   x = StateElement(
      numpy.array([[0, 0.1], [-0.5, 0.8]], dtype=numpy.float64),
      cont_space,
      out_of_bounds_mode="warning",
   )
   # to use numpy.amax, just cast x to an array with x.view(numpy.ndarray)
   max = numpy.amax(x.view(numpy.ndarray))

If you want to retain space information, there is another way, which depends on which type of Numpy function you want to use.

NumPy universal functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`Numpy universal functions <https://numpy.org/doc/stable/reference/ufuncs.html>`_ operate on ndarrays element-by-element, and support broad- and type-casting. A list of these functions is found `here <https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs>`_. All universal functions should work on StateElement. 

For example, addition is supported:

.. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
   :language: python
   :linenos:
   :start-after: [start-stateelement-ufunc]
   :end-before: [end-stateelement-ufunc]

.. note:: 

   Universal function will work the same way as with regular numpy arrays, but the end result is processed according to the ``"out_of_bounds_mode"`` of the ``StateElement``. Make sure to select a proper ``"out_of_bounds_mode"``.

.. warning::

   Not all functions are universal functions. For example ``iadd`` (in-place add), called when doing e.g. ``x+=1`` is not.


Using NumPy \_\_array_function\_\_ dispatching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For NumPy functions that are not universal functions, you can use NumPy's \_\_array_function\_\_ dispatching mechanism, see `Numpy docs <https://numpy.org/doc/stable/user/basics.dispatch.html>`_. In that case, you actually have to write the function. Once it's written, you can wrap it up in an ``@implements`` decorator, and then it's accessible to all StateElements. In fact, you can even submit a PR to CoopIHC's Github and add it permanently for others to use.
An example is provided below. Of course it helps to know how NumPy works.


.. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
   :language: python
   :linenos:
   :start-after: [start-stateelement-array-function]
   :end-before: [end-stateelement-array-function]




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

