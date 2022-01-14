.. CoopIHC documentation master file, created by
   sphinx-quickstart on Fri Apr  9 10:33:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.




Welcome to CoopIHC's documentation!
==============================================

.. warning:: 

    This is a version that is not ready for realease yet. Documentation  may be outdated in some places. Please contact me directly or raise an issue if appropriate.


*CoopIHC*, pronounced 'kopik', is a Python module that provides a common basis for describing **computational Human Computer Interaction (HCI)** contexts, mostly targeted at expressing models of users and intelligent assistants.

1. It provides a common conceptual and practical reference, which facilitates reusing and helps extending other researcher's work. Some examples that use *CoopIHC* can be found in the `CoopIHC-Zoo <https://jgori-ouistiti.github.io/CoopIHC-zoo/>`_. These full examples, or parts thereof, can be re-used easily by a *CoopIHC* end-user.
2. It can help design intelligent assistants. For example, you can wrap a CoopIHC :py:class:`Bundle<../coopihc.bundle.Bundle.Bundle>` into an environment that is compatible with `gym <https://gym.openai.com/>`_ and use off-the-shelf Deep Reinforcement Learning algorithms to train a policy for an intelligent assistant.
3. It provides modeling help. Currently, some checks are provided to ensure model paramters are correctly identifiable, see `CoopIHC-ModelChekcs <https://github.com/christophajohns/CoopIHC-ModelChecks/>`_



The philosophy of *CoopIHC* is to separate interactive systems into three components:

1. A **task**,
2. A **user**, which is either a real user *or* a synthetic user model,
3. An **assistant** agent which helps the user accomplish its task.

and **bundling** them back together using a so-called :py:mod:`Bundle <../coopihc.Bundle.Bundle>`. Bundles can be used for many different cases:

* Evaluate user (synthetic or real) coupled with an intelligent assistant,
* Train a user model to obtain a realistic synthetic user model,
* Train an intelligent assistant given some synthetic user model,
* Jointly train an intelligent assistant and a synthetic user model ...
* ... and more



*CoopIHC* builds on a two-agent interaction model, see the :doc:`Interaction Model<guide/interaction_model>` and :doc:`Terminology<guide/terminology>`.




.. toctree::
    :maxdepth: 1
    :caption: Tutorial

    guide/quickstart
    guide/more_complex_example
    guide/modularity

.. toctree::
    :maxdepth: 1
    :caption: Examples



.. toctree::
    :maxdepth: 1
    :caption: User Guide

    guide/interaction_model
    guide/space
    guide/stateelement
    guide/state
    guide/tasks
    guide/agents
    guide/policy
    guide/observation_engine
    guide/inference_engine
    guide/bundles
    guide/user_modeling
    guide/wrappers
    guide/repository


   
.. toctree::
    :maxdepth: 1
    :caption: See also

	Home page <self>
	API reference <_autosummary/coopihc>
    guide/terminology



Known Caveats
=====================

1. In place additions e.g. ``self.state['x'] += 1`` work, but do not trigger the expected ``out_of_bounds_mode`` behavior. in short, the reason for that is that in place addition calls ``__iadd__`` which is not a Numpy ``__ufunc__``. There are several workarounds possible. One based on the ``@implements`` mechanism described in the ``StateElement`` page which would fix the problem for everyone. Another is simply to do something like ``self.state['x'] = self.state['x'] + 1``




Indices
===========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
