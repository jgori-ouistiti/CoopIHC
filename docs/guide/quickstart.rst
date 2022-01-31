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

from within the folder. This will install *CoopIHC* in editable mode (basically equivalent to ``python3 -m pip install -e .``), together with all its dependencies.

Interaction Model
-------------------

*CoopIHC* builds on a :doc:`sequential two agent decision-making model<./interaction_model>`. You should read through the model to get a grasp of what each component does.


High-level view of CoopIHC code
--------------------------------

At a high level, your CoopIHC code will usually look like this

.. literalinclude:: ../../coopihc/examples/basic_examples/bundle_examples.py
   :language: python
   :linenos:
   :start-after: [start-highlevel-code]
   :end-before: [end-highlevel-code]


You will usually define a task, a user, an assistant, and bundle them together. You can then play several rounds of interaction until the game ends, and based on the collected data, you can do something.

States
------------
The interaction model uses the concept of states, for which *CoopIHC* introduces the ``Space`` ``StateElement`` and ``State`` objects. States can hold StateElements and can be nested. In the example below, a super-state is defined using a State. This super-state is itself defined by two substates. Each of those two substates holds a StateElement, which is a combination of a value and a space.

.. literalinclude:: ../../coopihc/examples/basic_examples/state_examples.py
   :language: python
   :linenos:
   :start-after: [start-state-example]
   :end-before: [end-state-example]


States and StatElements subclass the built-in dictionary and the well-known NumPy arrays respectively. This means that interacting with these objects should prove relatively familiar. To find out more about this and for extra details, go to :doc:`Space<space>`, :doc:`StateElement<stateelement>` and :doc:`State<state>`.


Tasks
--------

.. include:: tasks.rst
    :start-after: .. start-quickstart-task
    :end-before: .. end-quickstart-task


Agents
------------------

.. include:: agents.rst
    :start-after: .. start-quickstart-agent
    :end-before: .. end-quickstart-agent

Policies
------------

.. include:: policy.rst
    :start-after: .. start-quickstart-policy
    :end-before: .. end-quickstart-policy

Observation Engines
---------------------

.. include:: observation_engine.rst
    :start-after: .. start-quickstart-obseng-intro
    :end-before: .. end-quickstart-obseng-intro

.. include:: observation_engine.rst
    :start-after: .. start-quickstart-obseng-subclass
    :end-before: .. end-quickstart-obseng-subclass


Inference Engines
--------------------

.. include:: inference_engine.rst
    :start-after: .. start-quickstart-infeng-intro
    :end-before: .. end-quickstart-infeng-intro


.. include:: inference_engine.rst
    :start-after: .. start-quickstart-infeng-subclass
    :end-before: .. end-quickstart-infeng-subclass




Bundles
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
 


