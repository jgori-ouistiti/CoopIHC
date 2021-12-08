.. CoopIHC documentation master file, created by
   sphinx-quickstart on Fri Apr  9 10:33:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.




Welcome to CoopIHC's documentation!
==============================================

.. warning:: 

    This is a version that is not ready for realease yet. Documentation  may be outdated in some places. Please contact me directly or raise an issue if appropriate.


*CoopIHC*, pronounced 'kopik', is a Python module that provides a common basis for describing **computational Human Computer Interaction (HCI)** contexts, mostly targeted at expressing models of users and intelligent assistants.

1. It provides a common conceptual and practical reference, which facilitates reusing and helps extending other researcher's work. Some examples that use *CoopIHC* can be found in the `CoopIHC-Zoo <https://jgori-ouistiti.github.io/CoopIHC-zoo/>`_. These full examples, or parts thereof, can be re-used easily by a *CoopIHC* user.
2. It can help design intelligent assistants. For example, you can wrap a CoopIHC :py:mod:`Bundle <../coopihc.bundle>` into an environment that is compatible with `gym <https://gym.openai.com/>`_ and use off-the-shelf Deep Reinforcement Learning algorithms to train a policy for an intelligent assistant.



The philosophy of *CoopIHC* is to separate interactive systems into three components:

1. A **task**,
2. A **user**, which is either a real user *or* a synthetic user model,
3. An **assistant** agent which helps the user accomplish its task.

and **bundling** them back together using so-called :py:mod:`Bundle <../coopihc.bundle>`. Different bundles exist for many different cases:

* Evaluate user (synthetic or real) coupled with an intelligent assistant,
* Train a user model to obtain a realistic synthetic user model,
* Train an intelligent assistant given some synthetic user model,
* Jointly train an intelligent assistant and a synthetic user model ...
* ... and more



*CoopIHC* builds on a two-agent interaction model, see the :doc:`Interaction Model<guide/interaction_model>` and :doc:`Terminology<guide/terminology>`.




*CoopIHC* User Guide
===============================================



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

    guide/terminology
    guide/interaction_model
    guide/space
    guide/policy
    guide/observation_engine
    guide/inference_engine
    guide/tasks
    guide/agents
    guide/bundles
    guide/user_modeling
    guide/wrappers

	





.. toctree::
    :maxdepth: 1
    :caption: What's next?

    roadmap


   
.. toctree::
	:hidden:	

	Home page <self>
	API reference <_autosummary/coopihc>


TODO list:
==============

* have a task possess an observation engine for cleaner separability between modules
* think about standardized logging capabilities
* provide test code to ensure the engines are working properly
* verify full arithmetic operations for StateElement
* make a mapping object for RuleObservationEngine, smoothen the ruleObservationEngine specification
* profile CoopIHC to see whether there are any bottlenecks (deepcopies are one)



Indices
===========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
