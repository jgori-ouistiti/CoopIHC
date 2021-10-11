from collections import OrderedDict
import numpy
import queue
from core.space import State
from core.helpers import hard_flatten
import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.linalg

# Base Inference Engine: does nothing but return the same state. Any new inference method can subclass InferenceEngine to have a buffer and add_observation method (required by the bundle)


class BaseInferenceEngine:
    def __init__(self, buffer_depth=1):
        self.buffer = None
        self.buffer_depth = buffer_depth
        self.render_flag = None
        self.ax = None

    def __content__(self):
        return self.__class__.__name__

    @property
    def observation(self):
        return self.buffer[-1]

    @property
    def state(self):
        return self.buffer[-1]["{}_state".format(self.host.role)]

    @property
    def action(self):
        return self.host.policy.action_state["action"]

    @property
    def unwrapped(self):
        return self

    def add_observation(self, observation):
        """add an observation  to a naive buffer.

        :param observation: verify type.
        """

        if self.buffer is None:
            self.buffer = []
        if len(self.buffer) < self.buffer_depth:
            self.buffer.append(observation)
        else:
            self.buffer = self.buffer[1:] + [observation]

    # https://stackoverflow.com/questions/1015307/python-bind-an-unbound-method
    def bind(self, func, as_name=None):
        # print("\n")
        # print(func, as_name)
        if as_name is None:
            as_name = func.__name__
        bound_method = func.__get__(self, self.__class__)
        setattr(self, as_name, bound_method)
        return bound_method

    def infer(self):
        """The main method of this class.

        Return the new value of the internal state of the agent, as well as the reward associated with inferring the . By default, this inference engine does nothing, and just returns the state.

        :return: new_internal_state (OrderedDict), reward (float)
        """
        # do something with information inside buffer

        if self.host.role == "user":
            try:
                return self.buffer[-1]["user_state"], 0
            except KeyError:
                return OrderedDict({}), 0
        elif self.host.role == "assistant":
            try:
                return self.buffer[-1]["assistant_state"], 0
            except KeyError:
                return OrderedDict({}), 0

    def reset(self):
        self.buffer = None

    def render(self, *args, **kwargs):
        mode = kwargs.get("mode")

        render_flag = False
        for r in self.render_tag:
            if r in mode:
                render_flag = True

        if render_flag:

            if "plot" in mode:
                ax = args[0]
                if self.ax is not None:
                    pass
                else:
                    self.ax = ax
                    self.ax.set_title(type(self).__name__)

            if "text" in mode:
                print(type(self).__name__)


# The usermodel is not updated with this assistant
class GoalInferenceWithUserPolicyGiven(BaseInferenceEngine):
    """An Inference Engine used by an assistant to infer the goal of an user.

    The inference is based on an user_model which has to be provided to this engine.

    :meta public:
    """

    def __init__(self, *args):
        super().__init__()
        try:
            self.attach_policy(args[0])
        except IndexError:
            self.user_policy_model = None

        self.render_tag = ["plot", "text"]

    def attach_policy(self, policy):
        if not policy.explicit_likelihood:
            print(
                "Warning: This inference engine requires a policy defined by an explicit likelihood"
            )
        print(
            "Attached policy {} to {}".format(policy, self.__class__.__name__)
        )
        self.user_policy_model = policy

    def attach_set_theta(self, set_theta):
        self.set_theta = set_theta

    def render(self, *args, **kwargs):

        mode = kwargs.get("mode")

        render_flag = False
        for r in self.render_tag:
            if r in mode:
                render_flag = True

        ## ----------------------------- Begin Helper functions
        def set_box(
            ax,
            pos,
            draw="k",
            fill=None,
            symbol=None,
            symbol_color=None,
            shortcut=None,
            box_width=1,
            boxheight=1,
            boxbottom=0,
        ):
            if shortcut == "void":
                draw = "k"
                fill = "#aaaaaa"
                symbol = None
            elif shortcut == "target":
                draw = "#96006c"
                fill = "#913979"
                symbol = "1"
                symbol_color = "k"
            elif shortcut == "goal":
                draw = "#009c08"
                fill = "#349439"
                symbol = "X"
                symbol_color = "k"
            elif shortcut == "position":
                draw = "#00189c"
                fill = "#6573bf"
                symbol = "X"
                symbol_color = "k"

            BOX_HW = box_width / 2
            _x = [pos - BOX_HW, pos + BOX_HW, pos + BOX_HW, pos - BOX_HW]
            _y = [
                boxbottom,
                boxbottom,
                boxbottom + boxheight,
                boxbottom + boxheight,
            ]
            x_cycle = _x + [_x[0]]
            y_cycle = _y + [_y[0]]
            if fill is not None:
                fill = ax.fill_between(_x[:2], _y[:2], _y[2:], color=fill)

            (draw,) = ax.plot(x_cycle, y_cycle, "-", color=draw, lw=2)
            symbol = None
            if symbol is not None:
                symbol = ax.plot(
                    pos, 0, color=symbol_color, marker=symbol, markersize=100
                )

            return draw, fill, symbol

        def draw_beliefs(ax):
            beliefs = hard_flatten(self.host.state["beliefs"]["values"])
            ticks = []
            ticklabels = []
            for i, b in enumerate(beliefs):
                draw, fill, symbol = set_box(
                    ax, 2 * i, shortcut="target", boxheight=b
                )
                ticks.append(2 * i)
                ticklabels.append(i)
            self.ax.set_xticks(ticks)
            self.ax.set_xticklabels(ticklabels, rotation=90)

        ## -------------------------- End Helper functions

        if "plot" in mode:
            ax = args[0]
            if self.ax is not None:
                title = self.ax.get_title()
                self.ax.clear()
                draw_beliefs(ax)
                ax.set_title(title)

            else:
                self.ax = ax
                draw_beliefs(ax)
                self.ax.set_title(type(self).__name__ + " beliefs")

        if "text" in mode:
            beliefs = hard_flatten(self.host.state["beliefs"]["values"])
            print("beliefs", beliefs)

    def infer(self):
        """Update the substate 'beliefs' from the internal state. Generate candidate observations for each potential target, evaluate its likelihood and update the prior to form the posterior. Normalize the posterior and return the new state.

        :return: new internal state (OrderedDict), reward associated with inferring (float)

        """

        if self.user_policy_model is None:
            raise RuntimeError(
                "This inference engine requires a likelihood-based model of an user policy to function."
            )

        observation = self.buffer[-1]
        state = observation["assistant_state"]
        old_beliefs = state["beliefs"]["values"][0].squeeze().tolist()
        user_action = observation["user_action"]["action"]

        print(old_beliefs)
        for nt, t in enumerate(self.set_theta):
            print(nt, t)
            # candidate_observation = copy.copy(observation)
            candidate_observation = copy.deepcopy(observation)
            for key, value in t.items():
                try:
                    candidate_observation[key[0]][key[1]] = value
                except KeyError:  # key[0] is not in observation
                    _state = State()
                    _state[key[1]] = value
                    candidate_observation[key[0]] = _state

            old_beliefs[nt] *= self.user_policy_model.compute_likelihood(
                user_action, candidate_observation
            )

        if sum(old_beliefs) == 0:
            print(
                "warning: beliefs sum up to 0 after updating. I'm resetting to uniform to continue behavior. You should check if the behavior model makes sense. Here are the latest results from the model"
            )
            old_beliefs = [1 for i in old_beliefs]
        new_beliefs = [i / sum(old_beliefs) for i in old_beliefs]
        state["beliefs"]["values"] = numpy.array(new_beliefs)
        return state, 0


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
        new_sigma = numpy.linalg.inv(
            (numpy.linalg.inv(oldsigma) + numpy.linalg.inv(v))
        )
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
            belief = self.host.state["belief"]["values"][0].squeeze()
            if dim == 1:
                belief = [belief[0], 0]
            axtask.plot(*belief, "r*")
            self.ax.set_title(type(self).__name__ + " beliefs")

        if "text" in mode:
            print(self.host.state["belief"]["values"])

    def draw_beliefs(self, ax, dim):
        mu, cov = self.host.state["belief"]["values"]
        if dim == 2:
            self.patch = self.confidence_ellipse(mu.squeeze(), cov, ax)
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
        ## See https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html for source. Computing eigenvalues directly should lead to code that is more readily understandable
        rho = numpy.sqrt(
            covariance[0, 1] ** 2 / covariance[0, 0] / covariance[1, 1]
        )
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


class ContinuousKalmanUpdate(BaseInferenceEngine):
    def __init__(self):
        super().__init__()
        self.fmd_flag = False
        self.K_flag = False

    def set_forward_model_dynamics(self, A, B, C):
        self.fmd_flag = True
        self.A = A
        self.B = B
        self.C = C

    def set_K(self, K):
        self.K_flag = True
        self.K = K

    def infer(self):
        if not self.fmd_flag:
            raise RuntimeError(
                "You have to set the forward model dynamics, by calling the set_forward_model_dynamics() method with inference engine {}  before using it".format(
                    type(self).__name__
                )
            )
        if not self.K_flag:
            raise RuntimeError(
                "You have to set the K Matrix, by calling the set_K() method with inference engine {}  before using it".format(
                    type(self).__name__
                )
            )
        observation = self.observation
        dy = observation["task_state"]["x"]["values"][0] * self.host.timestep

        if isinstance(dy, list):
            dy = dy[0]
        if not isinstance(dy, numpy.ndarray):
            raise TypeError(
                "Substate Xhat of {} is expected to be of type numpy.ndarray".format(
                    type(self.host).__name__
                )
            )

        state = observation["{}_state".format(self.host.role)]
        u = self.action["values"][0]

        xhat = state["xhat"]["values"][0]

        xhat = xhat.reshape(-1, 1)
        u = u.reshape(-1, 1)
        deltaxhat = (
            self.A @ xhat + self.B @ u
        ) * self.host.timestep + self.K @ (
            dy - self.C @ xhat * self.host.timestep
        )
        xhat += deltaxhat
        state["xhat"]["values"] = xhat

        # Here, we use the classical definition of rewards in the LQG setup, but this requires having the true value of the state. This may or may not realistic...
        # ====================== Rewards ===============

        x = self.host.bundle.task.state["x"]["values"][0]
        reward = (x - xhat).T @ self.host.U @ (x - xhat)

        return state, reward


# ================ Examples ============


class ExampleInferenceEngine(BaseInferenceEngine):
    def infer(self):
        return self.state, 0
