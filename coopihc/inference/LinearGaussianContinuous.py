import numpy
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


class LinearGaussianContinuous(BaseInferenceEngine):
    """LinearGaussianContinuous

    An Inference Engine that handles a continuous Gaussian Belief. It assumes a Gaussian prior and a Gaussian likelihood. ---- Currently the covariance matrix for the likelihood is assumed to be contained by the host as self.Sigma. Maybe change this ----

    The mean and covariance matrices of Belief are stored in the substates 'MuBelief' and 'SigmaBelief'.

    :param likelihood_binding: function which computes the likelihood of each future observation
    :type likelihood_binding: function
    """

    def __init__(self, likelihood_binding, *args, **kwargs):

        super().__init__()
        self.render_tag = ["text", "plot"]

        self.bind(likelihood_binding, "provide_likelihood")

    def provide_likelihood(self):
        raise NotImplementedError(
            "You should bind a method named 'provide_likelihood' to this inference engine"
        )

    def infer(self):
        """infer

        Update the Gaussian beliefs: assuming a Gaussian noisy observation model:

        .. math::

            \\begin{align}
            p(y|x) \\sim \\mathcal{N}(x, \\Sigma_0)
            \\end{align}

        and with a Gaussian prior

        .. math::

            \\begin{align}
            p(x(t-1)) \\sim \mathcal{N}(\\mu(t-1), \\Sigma(t-1))
            \\end{align}

        we have that 

        .. math::

            \\begin{align}
            p(x(t) | y, x(t-1)) \\sim \\mathcal{N}(\\Sigma(t) \\left[ \\Sigma_0^{-1}y + \\Sigma(t-1) \\mu(t-1) \\right], \\Sigma(t)) \\\\
            \\Sigma(t) = (\\Sigma_0^{-1} + \\Sigma(t-1)^{-1})^{-1}
            \\end{align}

        :return: (new internal state, reward)
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, float)
        """
        observation = self.buffer[-1]
        print('"\n=======')
        print(dict.__repr__(observation))
        print(observation)
        if self.host.role == "user":
            state = observation["user_state"]
        else:
            state = observation["assistant_state"]

        if self.provide_likelihood is None:
            print(
                "Please call attach_yms() method before. You have to specify which components of the states constitute the observation that is used to update the beliefs."
            )
        else:
            y, v = self.provide_likelihood()

        oldmu = state["belief-mu"]
        oldsigma = state["belief-sigma"]
        new_sigma = numpy.linalg.inv((numpy.linalg.inv(oldsigma) + numpy.linalg.inv(v)))
        newmu = new_sigma @ (
            numpy.linalg.inv(v) @ y.T + numpy.linalg.inv(oldsigma) @ oldmu.T
        )
        state["belief-mu"][:] = newmu
        state["belief-sigma"][:, :] = new_sigma

        return state, 0

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
