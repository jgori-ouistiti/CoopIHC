.. state:


State
------------
States are the higher level containers used in *CoopIHC*, that can contain either StateElements or other States. They derive from Python's built-in dictionnary, and most of the dictionnary syntax and methods apply.

+ **Defining a State** is straightforward

   .. literalinclude:: ../../coopihc/examples/basic_examples/state_examples.py
      :language: python
      :linenos:
      :start-after: [start-state-example]
      :end-before: [end-state-example]

+ States can be **initialized** to a random values (or forced, like StateElements). To use the forced reset mechanism you have to provide a reset dictionnary, whose structure is identical to the state, see below. Substates which are not in the dictionnary are reset by random sampling.

   .. literalinclude:: ../../coopihc/examples/basic_examples/state_examples.py
      :language: python
      :linenos:
      :start-after: [start-state-reset]
      :end-before: [end-state-reset]

+ States can also be **filtered** by providing a dictionnary of items that you would like to retain, where each value is an index or slice that indicates which component of the StateElement you would like to filter out.

.. literalinclude:: ../../coopihc/examples/basic_examples/state_examples.py
   :language: python
   :linenos:
   :start-after: [start-state-filter]
   :end-before: [end-state-filter]

+ States can also be **serialized** to a a dictionnary

.. literalinclude:: ../../coopihc/examples/basic_examples/state_examples.py
   :language: python
   :linenos:
   :start-after: [start-state-serialize]
   :end-before: [end-state-serialize]

+ States can also be accessed with the dot notation:

.. code-block:: python

   state["sub1"] == state.sub1