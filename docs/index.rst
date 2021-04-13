.. interaction-agents documentation master file, created by
   sphinx-quickstart on Fri Apr  9 10:33:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to interaction-agents's documentation!
==============================================

A Python module that provides standardization of computational HCI models.

Overview
============

This is a version that is not ready for realease yet. Documentation is here solely for my own purpose.



Learn how to use *interaction-agents* quickly:
===============================================

.. toctree::
   :maxdepth: 2
   :caption: User Guide:


Documentation of *interaction-agents* code:
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

Indices
===========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
