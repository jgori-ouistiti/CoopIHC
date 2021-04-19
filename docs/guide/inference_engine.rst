.. inference_engine:

The Inference Engines
========================

The internal states of operators and assistants are expected to evolve over time, namely because both of them are learning from their observations during a run or several runs.

To account for this, *interaction-agents* provides an inference engine, which updates agent's internal states from their observations.

All inference engines are obtained by subclassing the base class ``InferenceEngine``. This inference engine provides a buffer and an API used by bundles to add observations to the engine.





The ``InferenceEngine``
---------------------------

The buffer that is maintained by the inference engine is a simple FIFO buffer, see below. The buffer depth parameter equals the number of observations that are stored. The example below has a ``buffer_depth=10``.

.. tikz:: Inference Engine Buffer
    :include: tikz/inference_engine.tikz
    :xscale: 100
    :align: center


Observations are added to the inference engine's buffer by the bundle, who calls the engine's ``add_observation`` method.

.. note::

    The flattened observations are stored in the buffer.

.. note::

    Default observation values for the buffer can be indicated using the ``init_value`` parameter

.. note::

    not tested, I think the flattening is not performed now.


List of inference engines (ongoing)
------------------------------------

* ``GoalInferenceWithOperatorModelGiven`` (GIWOMG) [link]:  An Inference Engine used by an assistant to infer the goal of an assistant. It assumes that the operator chooses as goal one of the targets of the task, stored in the 'Targets' substate of the task. It is also assumed that the assistant has an internal state 'Beliefs'. The inference is based on a discrete Bayes update, where the likelihood comes from an operator_model which has to be provided to this engine.



* ``ContinuousGaussian`` (CG) [link]: An Inference Engine that handles a multidimensional Gaussian belief. It assumes a Gaussian prior and a Gaussian likelihood. The mean and covariance matrices of Belief are stored in the substates 'MuBelief' and 'SigmaBelief'.

.. note::

    Currently the covariance matrix for the likelihood is assumed to be contained by the host as self.Sigma. Change this.

The following table summarizes the inference engines implemented.


======= ==============  ==========  ======  ===================================
Engine      Discrete    Continuous  Method   operator model has to be provided?
======= ==============  ==========  ======  ===================================
GIWOMG          ✔️                   Bayes                  ✔️
CG                          ✔️          ✔️                 (✔️)
======= ==============  ==========  ======  ===================================
