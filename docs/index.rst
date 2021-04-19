.. interaction-agents documentation master file, created by
   sphinx-quickstart on Fri Apr  9 10:33:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Caution
============

This is a version that is not ready for realease yet. Documentation is here solely for my own purpose.

Welcome to interaction-agents's documentation!
==============================================


*Interaction-agents* is a Python module that

1. Provides standardization of computational HCI models, and
2. Helps design intelligent user interfaces (IUIs).


The main idea of *interaction-agents* is to separate interactive systems into three components:

1. A task,
2. A user model, called the **operator**
3. A tool, called the **assistant**.

and **bundling** them back together. Different **bundles** are proposed, depending on the use case:

* Evaluate user and interface models,
* Train a user model for a given interface,
* Find the best interface given a user model,
* Jointly train interface and user models, to model adaptation ...
* ... and more

.. note::

    Another benefit of this separation is that it facilitates implementations, comparisons and evaluations by proposing a standardization of computational models in HCI. This will likely foster sharing/re-use of HCI models across researchers, as has been the case with other communities.




*Interaction-agents* builds on a two-agent interaction model, see [link] for the terminology used in this module.




Learn how to use *interaction-agents*:
===============================================



.. toctree::
  :maxdepth: 2
  :caption: Tutorial

  guide/terminology

.. toctree::
    :maxdepth: 2
    :caption: User Guide

    guide/interaction_model
    guide/design
    guide/agents
    guide/tasks
    guide/bundles
    guide/wrappers
    guide/observation_engine
    guide/inference_engine
    guide/operator_model


Documentation of *interaction-agents*:
============================================


.. toctree::
    :maxdepth: 2
    :caption: Core Module. This module is the core module of interaction-agents. Any new module should build heavily on existing classes, by subclassing them.

    modules/core/baseagents
    modules/core/interactiontask
    modules/core/bundle
    modules/core/inference
    modules/core/models
    modules/core/observation

.. toctree::
    :maxdepth: 2
    :caption: Pointing Module. This module is used to model pointing tasks.

    modules/pointing/operators
    modules/pointing/envs
    modules/pointing/assistants

.. toctree::
    :maxdepth: 2
    :caption: Eye Module. This module is used to model eye movements of human observers.

    modules/eye/operators
    modules/eye/envs
    modules/eye/noise


TODO list:
==============

1. Fix render signature
2. change all agent.observation to agent.observation_engine.observation
3. check if game_state_indices is still used
4. Make the RuleObservationEngine work again with slices/ other substate indicators
5. Make the process observation engine
6. Verify that the buffer works in the inference_engine
7. FIx note Continuous Gaussian inference engine

Indices
===========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
