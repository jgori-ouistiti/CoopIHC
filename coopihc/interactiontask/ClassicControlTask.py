import numpy
import copy

from coopihc.helpers import flatten
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.interactiontask.InteractionTask import InteractionTask


class ClassicControlTask(InteractionTask):
    """ClassicControlTask 

    A task used for a classic control setting with signal dependent and independent noise. You can account for control-dependent noise with an appropriate noise model in the policy or the observation engine.

    The task has a state x(.) which evolves according to 

    .. math ::

        \\begin{align}
            x(+.) = Ax(.) + Bu(.) + Fx(.).\\beta + G.\\omega + Hu(.)d\\gamma \\\\
        \\end{align}

    for "timespace=discrete" and

    .. math ::

        \\begin{align}
            x(+.) = (Ax(.) + Bu(.))dt + Fx(.).\\beta + G.\\omega \\\\
        \\end{align}

    for "timespace=continuous".

    where :math:``u(.)`` is the user action. The task is finised when the first component x[0,0] is close enough to 0. Currently this is implemented as the condition ``abs(x[0, 0]) <= 0.01``.

    where :math:`\\beta, \\omega \\sim \\mathcal{N}(0, \\sqrt{dt})` are Wiener processes.

    A and B may represent continuous or discrete dynamics. A conversion is implictly made following the value of discrete_dynamics keyword:

    .. math ::

        \\begin{align}
            A_c = \\frac{1}{dt} (A - I) \\\\
            B_c = B \\frac{1}{dt}
        \\end{align}

    .. math ::

        \\begin{align}
            A_d = I + dt \cdot{} A \\\\
            B_d = dt \cdot{} B
        \\end{align}



    :param timestep: dt
    :type timestep: float
    :param A: Passive dynamics
    :type A: numpy.ndarray
    :param B: Response to command
    :type B: numpy.ndarray
    :param F: signal dependent noise, defaults to None
    :type F: numpy.ndarray, optional
    :param G: independent noise, defaults to None
    :type G: numpy.ndarray, optional
    :param H: control-dependent noise, defaults to None
    :type H: numpy.ndarray, optional
    :param discrete_dynamics: whether A and B are continuous or discrete, defaults to True
    :type discrete_dynamics: bool, optional
    :param noise: whether to include noise, defaults to "on"
    :type noise: str, optional
    :param timespace: if the task is modeled as discrete or continuous, defaults to "discrete"
    :type noise: str, optional
    """

    def __init__(
        self,
        timestep,
        A,
        B,
        *args,
        F=None,
        G=None,
        H=None,
        discrete_dynamics=True,
        noise="on",
        timespace="discrete",
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.dim = max(A.shape)
        self.state = State()
        self.state["x"] = StateElement(
            numpy.zeros((self.dim, 1)),
            Space(
                [
                    -numpy.ones((self.dim, 1)) * numpy.inf,
                    numpy.ones((self.dim, 1)) * numpy.inf,
                ],
                "continuous",
            ),
        )
        self.state_last_x = copy.copy(self.state["x"])
        self.timestep = timestep

        if F is None:
            self.F = numpy.zeros(A.shape)
        else:
            self.F = F
        if G is None:
            self.G = numpy.zeros(A.shape)
        else:
            self.G = G
        if H is None:
            self.H = numpy.zeros(B.shape)
        # Convert dynamics between discrete and continuous.
        if discrete_dynamics:
            self.A_d = A
            self.B_d = B
            # Euler method
            self.A_c = 1 / timestep * (A - numpy.eye(A.shape[0]))
            self.B_c = B / timestep
        else:
            self.A_c = A
            self.B_c = B
            # Euler Method
            self.A_d = numpy.eye(A.shape[0]) + timestep * A
            self.B_d = timestep * B

        self.noise = noise
        self.timespace = timespace

    def finit(self):
        """finit

        Define whether to use continuous or discrete representation for A and B
        """
        if self.timespace == "continuous":
            self.A = self.A_c
            self.B = self.B_c
        else:
            self.A = self.A_d
            self.B = self.B_d

    def reset(self, dic=None):
        """Force all substates except the first to be null.

        Force all substates except the first to be null. Also stores the last state as an attribute (for rendering).


        :param dic: reset_dic, see :py:class:``InteractionTask <coopihc.interactiontask.InteractionTask.InteractionTask>``, defaults to None
        :type dic: dictionnary, optional
        """
        # Force zero velocity

        self.state["x"] *= numpy.array([1] + [0 for i in range(self.dim - 1)]).reshape(
            (-1, 1)
        )

        self.state_last_x = copy.copy(self.state["x"])

    def user_step(self, *args, **kwargs):
        """user step

        Takes the state from x(.) to x(+.) according to

        .. math ::

            \\begin{align}
                x(+.) = Ax(.) + Bu(.) + Fx(.).\\beta + G.\\omega \\\\
            \\end{align}

        """
        is_done = False
        # Call super for counters

        # For readability
        A, B, F, G, H = self.A, self.B, self.F, self.G, self.H

        _u = self.user_action.view(numpy.ndarray)
        _x = self.state["x"].view(numpy.ndarray)

        # Generate noise samples
        if self.noise == "on":
            beta, gamma = numpy.random.normal(0, numpy.sqrt(self.timestep), (2, 1))
            omega = numpy.random.normal(0, numpy.sqrt(self.timestep), (self.dim, 1))
        else:
            beta, gamma = numpy.random.normal(0, 0, (2, 1))
            omega = numpy.random.normal(0, 0, (self.dim, 1))

        # Store last_x for render
        self.state_last_x = copy.copy(self.state["x"])
        # Deterministic update + State dependent noise + independent noise
        if self.timespace == "discrete":
            _x = (A @ _x + B * _u) + F @ _x * beta + G @ omega + H * _u * gamma
        else:
            _x += (
                (A @ _x + B * _u) * self.timestep
                + F @ _x * beta
                + G @ omega
                + H * _u * gamma
            )

        self.state["x"][:] = _x
        if abs(_x[0, 0]) <= 0.01:
            is_done = True

        return self.state, 0, is_done

    def assistant_step(self, *args, **kwargs):
        """assistant_step"""
        return self.state, 0, False

    def render(self, *args, **kwargs):
        """render

        Text mode: print task state

        plot mode: Dynamically update axes with state trajectories.
        """
        mode = kwargs.get("mode")
        if mode is None:
            mode = "text"

        if "text" in mode:
            print("state")
            print(self.state["x"])
        if "plot" in mode:
            axtask, axuser, axassistant = args[:3]
            if self.ax is not None:
                self.draw()
                if self.turn_number == 3:
                    self.ax.legend(
                        handles=[self.axes[i].lines[0] for i in range(self.dim)]
                    )
            else:
                self.color = ["b", "g", "r", "c", "m", "y", "k"]
                self.labels = ["x[{:d}]".format(i) for i in range(self.dim)]
                self.axes = [axtask]
                self.ax = axtask
                for i in range(self.dim - 1):
                    self.axes.append(self.ax.twinx())

                for i, ax in enumerate(self.axes):
                    # ax.yaxis.label.set_color(self.color[i])
                    ax.tick_params(axis="y", colors=self.color[i])

                self.draw()

    def draw(self):
        if (self.state_last_x == self.state["x"]).all():
            pass
        else:
            for i in range(self.dim):
                self.axes[i].plot(
                    [
                        ((self.round_number - 1) / 2 - 1) * self.timestep,
                        (self.round_number - 1) / 2 * self.timestep,
                    ],
                    flatten(
                        [
                            self.state_last_x[i, 0].tolist(),
                            self.state["x"][i, 0].tolist(),
                        ]
                    ),
                    "-",
                    color=self.color[i],
                    label=self.labels[i],
                )

        return
