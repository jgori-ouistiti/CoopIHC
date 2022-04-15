.. parameters:


Parameters
------------
Sometimes, we need to store values that are shared across components, that don't need to be expressed as states. For example, you may want to fix a number of items, or the value of a quantity throughout. You could pass all these parameters to each component's constructor, but that would be a bit verbose. To make things easier, you can use *CoopIHC* ``parameters``. The example below shows how these work:

.. literalinclude:: ../../coopihc/examples/basic_examples/parameters_example.py
    :language: python
    :linenos:
    :start-after: [start-parameters-example]
    :end-before: [end-parameters-example]