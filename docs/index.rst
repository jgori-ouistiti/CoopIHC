.. CoopIHC documentation master file, created by
   sphinx-quickstart on Fri Apr  9 10:33:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Caution
============

This is a version that is not ready for realease yet. Documentation  is likely outdated. Please contact me if you have any questions.

Welcome to CoopIHC's documentation!
==============================================


*CoopIHC*, pronounced 'kopik', is a Python module that provides a common basis for describing computational Human Computer Interaction (HCI) contexts, mostly targeted at expressing models of users and intelligent assistants.

1. It provides a common conceptual and practical reference, which facilitates reusing and extending other researcher's work
2. It can help design intelligent assistants by translating an interactive context into a problem that can be solved (via other methods).


The main idea of *CoopIHC* is to separate interactive systems into three components:

1. A **task**,
2. A **user** (which may be a real user of a synthetic user model),
3. An assisting agent, the so-called **assistant**.

and **bundling** them back together. Different **bundles** are proposed, depending on the use case:

* Evaluate user and interface models,
* Train a user model for a given interface,
* Find the best interface given a user model,
* Jointly train interface and user models, to model adaptation ...
* ... and more



*CoopIHC* builds on a two-agent interaction model, see the :doc:`Terminology<guide/terminology>`.




Learn how to use *CoopIHC*:
===============================================



.. toctree::
    :maxdepth: 1
    :caption: Tutorial

    guide/quickstart
    guide/more_complex_example
    guide/modularity
    guide/training
    guide/external_components

.. toctree::
    :maxdepth: 1
    :caption: Examples

    guide/users


.. toctree::
    :maxdepth: 1
    :caption: User Guide

    guide/interaction_model
    guide/spaces
    guide/agents
    guide/tasks
    guide/bundles
    guide/wrappers
    guide/observation_engine
    guide/inference_engine
    guide/user_modeling



.. toctree:
    :caption: Miscellaneous
    guide/terminology


.. toctree::
    :maxdepth: 1
    :caption: What's next?

    roadmap


.. _modules-label:

List of Modules in *CoopIHC*
============================================


.. toctree::
    :maxdepth: 2
    :caption: Core module of CoopIHC. Any new module should build heavily on existing classes, by subclassing them.


    modules/coopihc/space
    modules/coopihc/observation
    modules/coopihc/inference
    modules/coopihc/policy
    modules/coopihc/agents
    modules/coopihc/interactiontask
    modules/coopihc/wsbundle

    
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
