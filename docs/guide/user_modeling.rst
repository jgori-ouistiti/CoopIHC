.. user_modeling:

User Modeling Facilitation
=============================

Facilitating user modeling with bundles. allow evaluation, training, parameter recovery etc.

To see how user modeling can be facilitated, we first build on a model presented `by Chen et al. <https://dl.acm.org/doi/fullHtml/10.1145/3411764.3445177>`. This will form the basis to which we will apply [things mentioned above].



Eye Fixation Model
--------------------
The model assumes the following:

* There is a screen with some target, that the eye should point towards. This constitutes the task
* The operator controls the eye; it does so by
    * Receiving noisy information about the target location
    * Handling beliefs about where the target might be, based on received information
    * Issuing the next position of the eye by selecting the most probable location based on handled beliefs
    * That action is corrupted by noise.

In all cases, the noise intensity is dependent on the distance between the target and the position.
* The task is done once the current eye position is close enough to the target, as determined by a threshold.



The code snippet below creates the needed bundle. First, we initialize a 1D pointing task, where the size and distance of the target are given by W and D.
We associate a ChenEye operator to the task, also in 1D. It gets as input the scaling linear coefficient for signal dependent noise (noise = coef * N(0,1)), for observation (perceptualnoise) as well as for moving the eye (oculomotornoise). These are bundled into a SinglePlayOperator bundle, which allows one to play as operator to experiment with policies. One could have also used a SinglePlayOperatorAuto bundle to directly evaluate the policy explained above, based on beliefs.

.. code-block:: python

    fitts_W = 4e-2
    fitts_D = 0.8
    perceptualnoise = 0.2
    oculomotornoise = 0.2
    task = ChenEyePointingTask(fitts_W, fitts_D, dimension = 1)
    operator = ChenEye(perceptualnoise, oculomotornoise, dimension = 1)
    bundle = SinglePlayOperator(task, operator)

    
Parameter Recovery
-------------------

In parameter recovery, a user model is tested on its ability to infer known parameters from an artificial experiment dataset.
If a model fails to do so, it is unlikely to be useful for inferring parameters from a dataset created with human users.

In the code below, we define a new kind of interaction task--in this case a risky choice task called `MultiBanditTask`--and test an operator class, namely Win-Stay-Lose-Switch (`WSLS`) for its ability to recover the single parameter `epsilon`.


.. code-block:: python

    # Imports
    from envs import MultiBanditTask
    from operators import WSLS

    from core.bundle import _DevelopOperator

    # Task parameters definition
    N = 2
    P = [0.5, 0.75]
    T = 100

    # Task definition
    multi_bandit_task = MultiBanditTask(N=N, P=P, T=T)

    # Operator definition
    wsls = WSLS(epsilon=0.1)


In order to test the model for parameter recovery, we need to use the `_DevelopOperator` bundle and pass it both the operator as well as the task.
In turn, it gives us access to a method called `test_parameter_recovery` that can be used to evaluate parameter recovery for the given operator class and task.

The `test_parameter_recovery` expects an argument called `parameter_fit_bounds` which is a `dict` containing the parameter name and its minimum and maximum value (i.e. its fit bounds).
These fit bounds will be used to create `n_simulations` artificial agents with random parameters within them.
Each of the artificial agents will execute the task to create simulated data which in turn can be used to infer the best-fit parameter values using an evaluation method, in this case BIC-score.

The result of calling `test_parameter_recovery` is a boolean whether all parameter correlations (i.e. Pearson's r for the correlation between the used and recovered parameter values) meet the specified `correlation_threshold` at the specified `significance_level`.

.. code-block:: python

    # Parameter fit bounds for operator
    wsls_parameter_fit_bounds = {"epsilon": (0., 1.)}

    # Population size
    N_SIMULATIONS = 20

    # Bundle defintion
    wsls_bundle = _DevelopOperator(task=multi_bandit_task, operator=wsls)

    # Parameter recovery check
    wsls_can_recover_parameters = wsls_bundle.test_parameter_recovery(parameter_fit_bounds=wsls_parameter_fit_bounds, correlation_threshold=0.6, significance_level=0.1, n_simulations=N_SIMULATIONS, plot=True)

    # Print result
    print(f"WSLS: Parameter recovery was {'successful' if wsls_can_recover_parameters else 'unsuccessful'}.")
