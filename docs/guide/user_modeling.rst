.. user_modeling:

User Modeling Facilitation
=============================

Facilitating user modeling with bundles. allow evaluation, training, parameter recovery etc.

To see how user modeling can be facilitated, we first build on a model presented `by Chen et al. <https://dl.acm.org/doi/fullHtml/10.1145/3411764.3445177>`. This will form the basis to which we will apply [things mentioned above].



Eye Fixation Model
--------------------
The model assumes the following:

* There is a screen with some target, that the eye should point towards. This constitutes the task
* The user controls the eye; it does so by
    * Receiving noisy information about the target location
    * Handling beliefs about where the target might be, based on received information
    * Issuing the next position of the eye by selecting the most probable location based on handled beliefs
    * That action is corrupted by noise.

In all cases, the noise intensity is dependent on the distance between the target and the position.
* The task is done once the current eye position is close enough to the target, as determined by a threshold.



The code snippet below creates the needed bundle. First, we initialize a 1D pointing task, where the size and distance of the target are given by W and D.
We associate a ChenEye user to the task, also in 1D. It gets as input the scaling linear coefficient for signal dependent noise (noise = coef * N(0,1)), for observation (perceptualnoise) as well as for moving the eye (oculomotornoise). These are bundled into a SinglePlayUser bundle, which allows one to play as user to experiment with policies. One could have also used a SinglePlayUserAuto bundle to directly evaluate the policy explained above, based on beliefs.

.. code-block:: python

    fitts_W = 4e-2
    fitts_D = 0.8
    perceptualnoise = 0.2
    oculomotornoise = 0.2
    task = ChenEyePointingTask(fitts_W, fitts_D, dimension = 1)
    user = ChenEye(perceptualnoise, oculomotornoise, dimension = 1)
    bundle = SinglePlayUser(task, user)