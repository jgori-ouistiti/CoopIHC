.. space:


Space
---------------
A ``Space`` is used by a ``StateElement`` to associate the value to a domain.  The interface for ``Space`` is explained in the :py:class:`API Reference<coopihc.base.Space.Space>`.

Defining spaces
^^^^^^^^^^^^^^^^^

Spaces can be either ``Numeric`` or ``CatSet``. Continuous spaces are only representable in the former, while discrete spaces can be represented in both objects. Specifically, ``CatSet`` is designed to hold categorical data, for which no ordering of values is possible.
Defining a space is done via the ``Space`` interface. To define ``Numeric`` spaces, you have to specify lower and upper bounds, while for ``CatSet`` you have to specify an array which holds all possible values.

   .. note::

      Performance for ``CatSet`` may be poor for large sets. 


 Some shortcuts also exist to make defining Spaces less cumbersome. Below are a few ways of defining a ``Space``

.. literalinclude:: ../../coopihc/examples/basic_examples/space_examples.py
   :language: python
   :linenos:
   :start-after: [start-space-def]
   :end-before: [end-space-def]


Mechanisms
^^^^^^^^^^^^

``Space`` comes with several mechanisms: 

1. check whether a value is contained in a space,
2. sample a value from a space,
3. Perform the cartesian product of spaces.


Values ``in`` space
************************
You can check if a value belongs to a space with Python's ``in`` operator:

.. literalinclude:: ../../coopihc/examples/basic_examples/space_examples.py
   :language: python
   :linenos:
   :start-after: [start-space-contains]
   :end-before: [end-space-contains]

Essentially, *CoopIHC* delegates this operation to Numpy.

Sampling from spaces
**************************
You can sample values from spaces using the ``sample()`` method:

.. literalinclude:: ../../coopihc/examples/basic_examples/space_examples.py
   :language: python
   :linenos:
   :start-after: [start-space-sample]
   :end-before: [end-space-sample]

You can provide a given seed for each ``Space`` via its ``seed`` keyword argument, for repeatable random experiments.




Cartesian Product
********************
You can compute the cartesian product of several spaces. For continuous spaces, a ``None`` value is used.

.. literalinclude:: ../../coopihc/examples/basic_examples/space_examples.py
   :language: python
   :linenos:
   :start-after: [start-space-cp]
   :end-before: [end-space-cp]


