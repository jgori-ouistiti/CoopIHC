.. quickstart:

Quick Start
===============


Installation
---------------

.. *CoopIHC* is currently available on `TestPyPI <https://test.pypi.org/project/coopihc/>`_, although this might not be the latest version. You can install the package using pip with the following command:

.. .. code-block:: python

..     python3 -m pip install --extra-index-url https://test.pypi.org/simple/ coopihc

*CoopIHC* is currently available on PyPI. You can install the package using pip with the following command:

.. code-block:: python

    python3 -m pip install coopihc

You can also build directly from the github repository to get the latest version. To do so, install poetry, and run 

.. code-block:: shell

    poetry install

from within the folder. This will install *CoopIHC* in editable mode (basically equivalent to ``python3 -m pip install -e .``), together with all its dependencies. You might need to download and install *CoopIHC-Zoo* as well for this to work.

Interaction Model
-------------------

*CoopIHC* builds on a :doc:`sequential two agent decision-making model<./interaction_model>`. You should read through the model to get a grasp of what each component does.


High-level view of *CoopIHC* code
-----------------------------------

At a high level, your *CoopIHC* code will usually look like this

.. literalinclude:: ../../coopihc/examples/basic_examples/bundle_examples.py
   :language: python
   :linenos:
   :start-after: [start-highlevel-code]
   :end-before: [end-highlevel-code]


You will usually define a task, a user, an assistant, and bundle them together. You can then play several rounds of interaction until the game ends, and based on the collected data, you can do something.

Quick-States
--------------
The interaction model uses the concept of states, a collection of useful variables for the system. In *CoopIHC* you define them via a ``State`` object. The states are containers that hold elements called ``StateElement``. A ``StateElement`` is a collection of a value and a ``Space``, its associated domain. A ``State`` may be nested and contain another ``State``.

In the example below, a super-state is defined using a State. This super-state is itself defined by two substates. Each of those two substates holds a ``StateElement``, defined here via shortcuts such as ``array_element``.

.. literalinclude:: ../../coopihc/examples/basic_examples/state_examples.py
   :language: python
   :linenos:
   :start-after: [start-state-example]
   :end-before: [end-state-example]


`:py:class:States<coopihc.base.State>` and `:py:class:StateElementss<coopihc.base.StateElements>` subclass respectively Python's built-in dictionary and NumPy arrays types. Interacting with these objects should prove relatively familiar to most Python users. To find out more about this and for extra details, go to :doc:`Space<space>`, :doc:`StateElement<stateelement>` and :doc:`State<state>`.


Quick-Tasks
------------

.. include:: tasks.rst
    :start-after: .. start-quickstart-task
    :end-before: .. end-quickstart-task


Quick-Agents
------------------

.. include:: agents.rst
    :start-after: .. start-quickstart-agent
    :end-before: .. end-quickstart-agent

Quick-Policies
----------------

.. include:: policy.rst
    :start-after: .. start-quickstart-policy
    :end-before: .. end-quickstart-policy

Quick-Observation Engines
---------------------------

.. include:: observation_engine.rst
    :start-after: .. start-quickstart-obseng-intro
    :end-before: .. end-quickstart-obseng-intro

.. include:: observation_engine.rst
    :start-after: .. start-quickstart-obseng-subclass
    :end-before: .. end-quickstart-obseng-subclass


Quick-Inference Engines
-------------------------

.. include:: inference_engine.rst
    :start-after: .. start-quickstart-infeng-intro
    :end-before: .. end-quickstart-infeng-intro


.. include:: inference_engine.rst
    :start-after: .. start-quickstart-infeng-subclass
    :end-before: .. end-quickstart-infeng-subclass




Quick-Bundles
---------------------

.. include:: bundles.rst
    :start-after: .. start-quickstart-bundle
    :end-before: .. end-quickstart-bundle


.. note::

    Bundles also handle joint rendering as well as other practical things. More details can be found on :doc:`Bundle's reference page <bundles>`.


An overview of *CoopIHC*
-----------------------------------------------------


1. Several implementations of user models, tasks and assistants exist in *CoopIHC*'s repository `CoopIHC-Zoo <https://github.com/jgori-ouistiti/CoopIHC-zoo>`_  
2. Several worked-out examples are given in this documentation. Those should give you a good idea about what can be done with *CoopIHC*.
 


