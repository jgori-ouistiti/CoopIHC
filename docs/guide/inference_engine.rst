.. inference_engine:

The Inference Engines
========================

.. start-quickstart-infeng-intro

Real-life agents have non-stationary policies. This gives them the ability to learn (infer parameters from observed data) and to adapt (change policy parameters based on observed data). As in observation engines, there might be a cost associated with making inferences:

    + Making an inference can be time costly.

    + Inferring may be rewarding; for example, because it is enjoyable.


*CoopIHC* provides a generic object called inference engines to updating internal states from observations. Although the name might suggest otherwise, these engines may use other mechanisms than statistical inference that update the internal state. To create a new inference engine, you can base it off an existing engine or subclass the ``BaseInferenceEngine``. 

.. end-quickstart-infeng-intro


Subclassing BaseInferenceEngine
------------------------------------------

.. start-quickstart-infeng-subclass

Essentially, the ``BaseInferenceEngine`` provides a simple first-in-first-out (FIFO) buffer that stores observations. When subclassing ``BaseInferenceEngine``, you simply have to redefine the ``infer`` method (by default, no inference is produced). An example is provided below, where the engine stores the last 5 observations. 

.. The example below has a ``buffer_depth=10``.

.. .. tikz:: Inference Engine Buffer
..     :include: tikz/inference_engine.tikz
..     :xscale: 100
..     :align: center


.. Observations are added to the inference engine's buffer by the bundle, who calls the engine's ``add_observation`` method.


.. literalinclude:: ../../coopihc/inference/ExampleInferenceEngine.py
   :language: python
   :linenos:
   :start-after: [start-infeng-subclass]
   :end-before: [end-infeng-subclass]

.. end-quickstart-infeng-subclass


Combining Engines -- CascadedInferenceEngine
----------------------------------------------
It is sometimes useful to use several inference engine in a row (e.g. because you want to use two engines that target a different substate). 

For this case, you can use the ``CascadedInferenceEngine``:

.. code-block::

    first_inference_engine = ProvideLikelihoodInferenceEngine(perceptualnoise)
    second_inference_engine = LinearGaussianContinuous()
    inference_engine = CascadedInferenceEngine(
        [first_inference_engine, second_inference_engine]
    )


Available Inference Engines
-------------------------------

GoalInferenceWithUserModelGiven
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An inference Engine used by an assistant to infer the 'goal' of a user.
The inference is based on a model of the user policy, which has to be provided to this engine.


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


LinearGaussianContinuous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An Inference Engine that handles a continuous Gaussian Belief. It assumes a Gaussian prior and a Gaussian likelihood.

- **Expectations of the engine**

    This inference engine expects the agent to have in its internal state:

        + The mean matrix of the belief, stored as 'belief-mu'
        + The covariance matrix of the belief, stored as 'belief-sigma'
        + The new observation, stored as 'y'
        + The covariance matrix associated with the observation, stored as 'Sigma_0'


- **Inference**

    This engine uses the observation to update the beliefs (which has been computed from previous observations).

    To do so, a Gaussian noisy observation model is assumed, where x is the latest mean matrix of the belief.

    .. math::

        \\begin{align}
        p(y|x) \\sim \\mathcal{N}(x, \\Sigma_0)
        \\end{align}


    If the initial prior (belief probability) is Gaussian as well, then the posterior will remain Gaussian (because we are only applying linear operations to Gaussians, Gaussianity is preserved). So the posterior after t-1 observations has the following form, where :math:`(\\mu(t-1), \\Sigma(t-1))` are respectively the mean and covariance matrices of the beliefs.

    .. math::

        \\begin{align}
        p(x(t-1)) \\sim \mathcal{N}(\\mu(t-1), \\Sigma(t-1))
        \\end{align}

    On each new observation, the mean and covariance matrices are updated like so:

    .. math::

        \\begin{align}
        p(x(t) | y, x(t-1)) \\sim \\mathcal{N}(\\Sigma(t) \\left[ \\Sigma_0^{-1}y + \\Sigma(t-1) \\mu(t-1) \\right], \\Sigma(t)) \\\\
        \\Sigma(t) = (\\Sigma_0^{-1} + \\Sigma(t-1)^{-1})^{-1}
        \\end{align}


- **Render**

    ---- plot mode:

    This engine will plot mean beliefs on the task axis and the covariance beliefs on the agent axis, plotted as confidence intervals (bars for 1D and ellipses for 2D).


- **Example files**

    coopihczoo.eye.users


ContinuousKalmanUpdate
^^^^^^^^^^^^^^^^^^^^^^^^

LQG update, not documented yet, see API Reference


