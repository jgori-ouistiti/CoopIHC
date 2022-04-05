.. bundles:

Bundles
==================

.. start-quickstart-bundle

Bundles are the objects that compose the three main components (task, user and assistant) into a game. It forms the joint state, collects the rewards and ensure synchronous sequential sequences of observations, inferences and actions of the two agents. 


.. image:: images/bundle.png
    :align: center
    :width: 800


They are useful because they allow you to orchestrate the interaction how you want it.


In most cases, there is no need to define a new Bundle, and you can straightaway use the standard existing ``Bundle``. For example, you can create a bundle and interact with it like so: 

.. literalinclude:: ../../coopihc/examples/basic_examples/bundle_examples.py
    :language: python
    :linenos:
    :start-after: [start-check-taskuser]
    :end-before: [end-check-taskuser]


.. end-quickstart-bundle


Overview of Bundle mechanisms
-------------------------------

The main API methods are the same as gym's, although their functionality is extended.

* ``reset``, which allows you to have control which components you reset and how you reset them.
* ``step``, which allows you to specify how agents select their actions, and how many turns to play .
* ``render``, which combines all rendering methods of the components.
* ``close``, which closes the bundle properly.


The following graphic explicits how the most important options work

.. image:: images/turns.png
    :align: center
    :width: 800


.. note::

    When stepping through a Bundle, you can pass user and assistant actions e.g. ``bundle.step(user_action = 1)``. Any action passed like this will take precedence over the agent's ``sample`` method.