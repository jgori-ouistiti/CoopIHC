.. rng:

Random Number Generators
=========================

Usually when dealing with stochastic processes, there comes a point where you want to set the seed used by the Random Number Generator (RNG). This may be for reproducing results by a colleague, forcing a deterministic behavior when debugging a code etc.

*CoopIHC* supports setting seeds for the following two sources:

    * The ``sample()`` method of ``Space``.
    * Other calls to a RNG, e.g., when implementing a stochastic transition function.


Seeding ``sample()``
-----------------------

To set the seed for all spaces, you can directly pass a ``seed`` keyword argument to ``Bundle``:

.. literalinclude:: ../../coopihc/examples/basic_examples/bundle_examples.py
    :language: python
    :linenos:
    :start-after: [start-bundle-seed]
    :end-before: [end-bundle-seed]

.. note::
    
    You can set seeds manually, by calling an object's ``_set_seed()`` method, or by passing a ``seedsequence`` keyword argument to the object's constructor. However, setting seeds via the bundle is the preferred way, because it ensures a very low risk of RNG collisions between the various spaces, see `NumPy's documentation for parallel RNG <https://numpy.org/doc/stable/reference/random/parallel.html>`_ 

Calling a RNG from within *CoopIHC* code
--------------------------------------------

You can call an RNG, which will use a spawn from the same seed that was used during the bundle initialization, by calling a *CoopIHC* component's ``_get_rng()`` method.

.. code-block::

    # Inside a *CoopIHC* component
    # generate a random number
    self.get_rng().random()