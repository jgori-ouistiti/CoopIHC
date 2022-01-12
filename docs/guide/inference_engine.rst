.. inference_engine:

The Inference Engines
========================

The internal states of users and assistants are expected to evolve over time, namely because both of them are learning from their observations during a run or several runs.

To account for this, *CoopIHC* provides an inference engine, which updates agent's internal states from their observations.

All inference engines are obtained by subclassing the base class ``BaseInferenceEngine``. 



The ``BaseInferenceEngine``
---------------------------

The buffer that is maintained by the base inference engine is a simple FIFO buffer, see below. The buffer depth parameter equals the number of observations that are stored. The example below has a ``buffer_depth=10``.

.. tikz:: Inference Engine Buffer
    :include: tikz/inference_engine.tikz
    :xscale: 100
    :align: center


Observations are added to the inference engine's buffer by the bundle, who calls the engine's ``add_observation`` method.


List of inference engines (ongoing)
------------------------------------

* ``GoalInferenceWithUserModelGiven`` (GIWUMG) [link]:  An Inference Engine used by an assistant to infer the goal of an assistant. It assumes that the user chooses as goal one of the targets of the task, stored in the 'Targets' substate of the task. It is also assumed that the assistant has an internal state 'Beliefs'. The inference is based on a discrete Bayes update, where the likelihood comes from an user_model which has to be provided to this engine.



* ``ContinuousGaussian`` (CG) [link]: An Inference Engine that handles a multidimensional Gaussian belief. It assumes a Gaussian prior and a Gaussian likelihood. The mean and covariance matrices of Belief are stored in the substates 'MuBelief' and 'SigmaBelief'.

.. note::

    Currently the covariance matrix for the likelihood is assumed to be contained by the host as self.Sigma. 

The following table summarizes the inference engines implemented.


======= ==============  ==========  ======  ===================================
Engine      Discrete    Continuous  Method   user model has to be provided?
======= ==============  ==========  ======  ===================================
GIWUMG          ✔️                   Bayes                  ✔️
CG                          ✔️       Bayes                 (✔️)
======= ==============  ==========  ======  ===================================

``GoalInferenceWithUserModelGiven`` (GIWUMG)
""""""""""""""""""""""""""""""""""""""""""""""""""""
Bayesian updating in the discrete case.
Computes for each target :math:`\theta` the associated posterior probability, given an observation :math:`x` and the last user action :math:`y`:

.. math::

    P(\Theta = \theta | X=x, Y=y) = \frac{p(Y = y | \Theta = \theta, X=x)}{\sum_{\Theta} p(Y=y|\Theta = \theta, X=x)} P(\Theta = \theta).

This inference engine expects the likelihood model :math:`p(Y = y | \Theta = \theta, X=x)` to be supplied:

.. code-block:: python

    # Define the likelihood model for the user policy
    # user_policy_model = XXX

    inference_engine = GoalInferenceWithUserPolicyGiven()
    # Attach it to the engine
    inference_engine.attach_policy(user_policy_model)

It also expects that the set of :math:`\theta`'s is supplied:

.. code-block:: python

    set_theta = [
        {
            ("user_state", "goal"): StateElement(
                t,
                discrete_space(numpy.array(list(range(self.bundle.task.gridsize)))),
            )
        }
        for t in self.bundle.task.state["targets"]
    ]

    inference_engine.attach_set_theta(set_theta)

You can find a full worked-out example in CoopIHC-Zoo's pointing module.


``ContinuousGaussian`` (CG)
"""""""""""""""""""""""""""""""
Bayesian updating, but in the continuous multivariate case. Assumes Gaussian likelihoods and priors.

With a likelihood of the form

.. math::

    p(y|x) \sim \mathcal{N}(x, \Sigma_0)

and with a Gaussian prior

.. math::

    p(x(t-1)) \sim \mathcal{N}(\mu(t-1), \Sigma(t-1))

computes the posterior as

.. math::

    p(x(t) | y, x(t-1)) \sim \mathcal{N}(\Sigma(t) \left[ \Sigma_0^{-1}y + \Sigma(t-1) \mu(t-1) \right], \Sigma(t)) \\
    \Sigma(t) = (\Sigma_0^{-1} + \Sigma(t-1)^{-1})^{-1}
