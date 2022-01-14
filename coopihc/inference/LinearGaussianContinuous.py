import numpy
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


class LinearGaussianContinuous(BaseInferenceEngine):
    """LinearGaussianContinuous

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


    """

    def __init__(self, *args, **kwargs):

        super().__init__()
        self.render_tag = ["text", "plot"]

    def infer(self, user_state=None):
        if user_state is None:
            observation = self.observation
            if self.host.role == "user":
                user_state = observation["user_state"]
            else:
                user_state = observation["assistant_state"]

        # Likelihood model
        y, v = user_state["y"].view(numpy.ndarray), user_state["Sigma_0"].view(
            numpy.ndarray
        )
        # Prior
        oldmu, oldsigma = user_state["belief-mu"].view(numpy.ndarray), user_state[
            "belief-sigma"
        ].view(numpy.ndarray)

        # Posterior
        new_sigma = numpy.linalg.inv((numpy.linalg.inv(oldsigma) + numpy.linalg.inv(v)))
        newmu = new_sigma @ (
            numpy.linalg.inv(v) @ y + numpy.linalg.inv(oldsigma) @ oldmu
        )
        user_state["belief-mu"][:] = newmu
        user_state["belief-sigma"][:, :] = new_sigma

        return user_state, 0

    def render(self, *args, **kwargs):
        """render

        Draws the beliefs (mean value and ellipsis or confidence intervals according to dimension).
        """
        mode = kwargs.get("mode")
        render_flag = False
        for r in self.render_tag:
            if r in mode:
                render_flag = True

        if "plot" in mode:
            axtask, axuser, axassistant = args[:3]
            if self.host.role == "user":
                ax = axuser
            else:
                ax = axassistant

            dim = self.host.dimension
            if self.ax is not None:
                pass
            else:
                self.ax = ax

            self.draw_beliefs(ax, dim)
            mean_belief = self.host.state["belief-mu"]
            if dim == 1:
                mean_belief = numpy.array([mean_belief.squeeze().tolist(), 0])

            axtask.plot(*mean_belief.squeeze().tolist(), "r*")
            self.ax.set_title(type(self).__name__ + " beliefs")

        if "text" in mode:
            print(
                "Belief: Mu = {}, Sigma = {}".format(
                    self.host.state["belief-mu"].view(numpy.ndarray),
                    self.host.state["belief-sigma"].view(numpy.ndarray),
                )
            )

    def draw_beliefs(self, ax, dim):
        """draw_beliefs

        Draw beliefs of dimension 'dim' on axis 'ax'.

        :param ax: axis
        :type ax: matplotlib object
        :param dim: dimension of data
        :type dim: int
        """
        mu, cov = self.host.state["belief-mu"], self.host.state["belief-sigma"]
        print(mu, cov)
        if dim == 2:
            self.patch = self.confidence_ellipse(mu, cov, ax)
        else:
            self.patch = self.confidence_interval(mu, cov, ax)

    def confidence_interval(self, mu, cov, ax, n_std=2.0, color="b", **kwargs):
        """confidence_interval

        Compute confidence interval. For the Gaussian case like here, this is straightforward

        :param mu: mean matrix
        :type mu: numpy.ndarray
        :param cov: covariance matrix
        :type cov: numpy.ndarray
        :param ax: axis
        :type ax: matplotlib object
        :param n_std: size of the confidence interval in std, defaults to 2.0
        :type n_std: float, optional
        :param color: color of the CI, defaults to "b"
        :type color: str, optional
        """
        vec = [
            (mu - 2 * numpy.sqrt(cov))
            .reshape(
                1,
            )
            .tolist(),
            (mu + 2 * numpy.sqrt(cov))
            .reshape(
                1,
            )
            .tolist(),
        ]
        ax.plot(
            vec,
            [
                -0.5 + 0.05 * self.host.bundle.round_number,
                -0.5 + 0.05 * self.host.bundle.round_number,
            ],
            "-",
            marker="|",
            markersize=10,
            color=color,
            lw=2,
            **kwargs
        )

    def confidence_ellipse(
        self,
        mu,
        covariance,
        ax,
        n_std=2.0,
        facecolor="#d1dcf0",
        edgecolor="b",
        **kwargs
    ):
        """confidence_ellipse

        Draw confidence ellipsis. See `Matplotlib documentation <https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html>`_ for source. Computing eigenvalues directly should lead to code that is more readily understandable.

        :param mu: mean matrix
        :type mu: numpy.ndarray
        :param cov: covariance matrix
        :type cov: numpy.ndarray
        :param ax: axis
        :type ax: matplotlib object
        :param n_std: size of the confidence interval in std, defaults to 2.0
        :type n_std: float, optional
        :param facecolor: fill color, defaults to "#d1dcf0"
        :type facecolor: str, optional
        :param edgecolor: frontier color, defaults to "b"
        :type edgecolor: str, optional
        """

        mu = mu.squeeze()

        rho = numpy.sqrt(covariance[0, 1] ** 2 / covariance[0, 0] / covariance[1, 1])
        ellipse_radius_x = numpy.sqrt(1 + rho)
        ellipse_radius_y = numpy.sqrt(1 - rho)

        ellipse1 = Ellipse(
            (0, 0),
            width=ellipse_radius_x * 2,
            height=ellipse_radius_y * 2,
            facecolor=facecolor,
            edgecolor=edgecolor,
            **kwargs
        )

        scale_x = numpy.sqrt(covariance[0, 0]) * n_std
        mean_x = mu[0]

        scale_y = numpy.sqrt(covariance[1, 1]) * n_std
        mean_y = mu[1]

        transf = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(mean_x, mean_y)
        )

        ellipse1.set_transform(transf + ax.transData)
        ax.add_patch(ellipse1)

        ellipse2 = Ellipse(
            (0, 0),
            width=ellipse_radius_x * 2,
            height=ellipse_radius_y * 2,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=0.3,
            **kwargs
        )
        n_std = n_std * 2
        scale_x = numpy.sqrt(covariance[0, 0]) * n_std
        mean_x = mu[0]

        scale_y = numpy.sqrt(covariance[1, 1]) * n_std
        mean_y = mu[1]

        transf = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(mean_x, mean_y)
        )

        ellipse2.set_transform(transf + ax.transData)
        ax.add_patch(ellipse2)

        return
