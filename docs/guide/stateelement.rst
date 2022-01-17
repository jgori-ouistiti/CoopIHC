.. stateelement:

StateElement
-----------------

A ``StateElement`` is a a combination of a value and a space. Under the hood, ``StateElement`` subclasses ``numpy.ndarray``; essentially, it adds a layer that checks whether the values are contained in the space (and what to do if not). As a result, many NumPy methods will work.

Instantiating a StateElement is straightforward.

.. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
   :language: python
   :linenos:
   :start-after: [start-stateelement-init]
   :end-before: [end-stateelement-init]


.. note::

   The examples above give the preferred input shape, but StateElement will consider the input as an array and try to viewcast input that does not match expected shape. That means that e.g. ``x = StateElement(2, discr_space, out_of_bounds_mode="error")`` is also considered valid input.



``StateElement`` has an ``out_of_bounds_mode`` keyword argument (defaults to 'warning') that specifies what to do when a value is not contained in the space:

* "error" --> raises a ``StateNotContainedError``
* "warning" --> warns with a ``StateNotContainedWarning``, but accepts the input
* "clip" --> clips the data to force it to belong to the space
* "silent" --> Values not in the space are accepted silently (behavior should become equivalent to numpy.ndarray). 
* "raw" --> No data transformation is applied. This is faster than the other options, because the preprocessing of input data is short-circuited. However, this provides no tolerance for misspecified input.

.. note:: 
   
   Broad/viewcasting and type casting are applied if necessary to the first four cases if the space has "contains" set to "soft", but never with "raw". 

Using NumPy functions
^^^^^^^^^^^^^^^^^^^^^^
There are several ways to use Numpy functions directly on StateElements. The easiest case is when you are okay with losing the information pertaining to space. In that case, you can just work with the array directly by casting the stateelement to an array:

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
`Numpy universal functions <https://numpy.org/doc/stable/reference/ufuncs.html>`_ are a collection of functions which operate on ndarrays element-by-element, and support broad- and type-casting. A list of these functions is found `here <https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs>`_. All universal functions should work on StateElement. 

For example, this example shows addition and equality to be supported, since both a universal functions.

.. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
   :language: python
   :linenos:
   :start-after: [start-stateelement-ufunc]
   :end-before: [end-stateelement-ufunc]

.. note:: 

   Universal function will work the same way as with regular numpy arrays, but the end result is processed according to the ``"out_of_bounds_mode"`` of the ``StateElement``. Make sure to select a proper ``"out_of_bounds_mode"``.

.. warning::

   Not all functions are universal functions. For example in-place addition (``numpy.iadd()``, called when doing e.g. ``x += 1``) is not.


Using NumPy \_\_array_function\_\_ dispatching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For NumPy functions that are not universal functions, you can apply the function directly on the StateElement. This will issue a ``NumpyFunctionNotHandledWarning``, and will call the numpy function on the StateElement cast as a numpy array. In practice, this is equivalent to the first solution ``numpy.amax(x.view(numpy.ndarray))``, except you didn't cast the StateElement yourself and therefore you are being warned.

.. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
   :language: python
   :linenos:
   :start-after: [start-stateelement-array-function-not-defined]
   :end-before: [end-stateelement-array-function-not-defined]

To retain the state information you can use NumPy's \_\_array_function\_\_ dispatching mechanism, see `Numpy docs <https://numpy.org/doc/stable/user/basics.dispatch.html>`_. In that case, you actually have to write the function. Once it's written, you can wrap it up in an ``@implements`` decorator, and then it's accessible to all StateElements. In fact, you can even submit a PR to CoopIHC's Github and add it permanently for others to use.
An example is provided below. Of course it helps to know how NumPy works.
The example below shows an imperefect implementation of ``amax``, that only works for continuous spaces and default amax arguments.

.. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
   :language: python
   :linenos:
   :start-after: [start-stateelement-array-function-define]
   :end-before: [end-stateelement-array-function-define]

Other mechanisms
^^^^^^^^^^^^^^^^^^^

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


+ **You can compare them**. This includes a "hard" comparison, which checks if spaces are equal.

   .. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
         :language: python
         :linenos:
         :start-after: [start-stateelement-equal]
         :end-before: [end-stateelement-equal]

+ **You can extract values with or without spaces**. Extracting the spaces together with the values can be done by a mechanism that abuses the slice notation.

   .. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
         :language: python
         :linenos:
         :start-after: [start-stateelement-getitem]
         :end-before: [end-stateelement-getitem]

+ **You can cast values from one space to the other**. This includes two modes of casting between discrete and continuous spaces.

   .. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
         :language: python
         :linenos:
         :start-after: [start-stateelement-cast]
         :end-before: [end-stateelement-cast]

.. + **You can cast values of a StateElement, to and from continuous to and from discrete spaces.**

..     .. literalinclude:: ../../coopihc/examples/simple_examples/stateelement_examples.py
..        :language: python
..        :linenos:
..        :start-after: [start-stateelement-cast]
..        :end-before: [end-stateelement-cast]


