import numpy
from coopihc.inference import BaseInferenceEngine
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


class LinearGaussianContinuous(BaseInferenceEngine):
    """An Inference Engine that handles a Gaussian Belief. It assumes a Gaussian prior and a Gaussian likelihood. ---- Currently the covariance matrix for the likelihood is assumed to be contained by the host as self.Sigma. Maybe change this ----

    The mean and covariance matrices of Belief are stored in the substates 'MuBelief' and 'SigmaBelief'.


    :meta public:
    """

    def __init__(self, likelihood_binding):
        super().__init__()
        self.render_tag = ["text", "plot"]

        self.bind(likelihood_binding, "provide_likelihood")

    def provide_likelihood(self):
        raise NotImplementedError(
            "You should bind a method named 'provide_likelihood' to this inference engine"
        )

    def infer(self):
        """Update the Gaussian beliefs, see XX for more information.

        :return: (OrderedDict) state, (float) 0

        :meta public:
        """
        observation = self.buffer[-1]
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

        oldmu, oldsigma = state["belief"]["values"]
        new_sigma = numpy.linalg.inv((numpy.linalg.inv(oldsigma) + numpy.linalg.inv(v)))
        newmu = new_sigma @ (
            numpy.linalg.inv(v) @ y.T + numpy.linalg.inv(oldsigma) @ oldmu.T
        )
        state["belief"]["values"] = [newmu, new_sigma]
        return state, 0

    def render(self, *args, **kwargs):
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
            mean_belief, std_belief = self.host.state["belief"]["values"]
            if dim == 1:
                mean_belief = numpy.array([mean_belief.squeeze().tolist(), 0])

            axtask.plot(*mean_belief.squeeze().tolist(), "r*")
            self.ax.set_title(type(self).__name__ + " beliefs")

        if "text" in mode:
            print(self.host.state["belief"]["values"])

    def draw_beliefs(self, ax, dim):
        mu, cov = self.host.state["belief"]["values"]
        print(mu, cov)
        if dim == 2:
            self.patch = self.confidence_ellipse(mu, cov, ax)
        else:
            self.patch = self.confidence_interval(mu, cov, ax)

    def confidence_interval(self, mu, cov, ax, n_std=2.0, color="b", **kwargs):
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
                -0.5 + 0.05 * self.host.bundle.task.round,
                -0.5 + 0.05 * self.host.bundle.task.round,
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
        """
        :meta private:
        """
        mu = mu.squeeze()
        ## See https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html for source. Computing eigenvalues directly should lead to code that is more readily understandable
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
