import numpy
import copy

from coopihc.helpers import flatten
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.interactiontask.InteractionTask import InteractionTask


class ClassicControlTask(InteractionTask):
    """ClassicControlTask 

    A task used for the classic control setting

    .. math ::

        \\begin{align}
            x(+) = Ax() + Bu() + Fx().\\beta + G.\\omega \\\\
        \\end{align}

    where :math:`\\beta, \\omega \\sim \\mathcal{N}(0, \\sqrt{dt})` are Wiener processes.

    A and B may represent continuous or discrete dynamics. A conversion is implictly made following the value of discrete_dynamics keyword:

    .. math ::

        \\begin{align}
            A_c = \\frac{1}{dt} (A - I) \\\\
            B_c = B \\frac{1}{dt}
        \\end{align}

    .. math ::

        \\begin{align}
            A_d = I + dt \cdot{} A
            B_d = dt \cdot{} B
        \\end{align}

    :param timestep: dt
    :type timestep: float
    :param A: Passive dynamics
    :type A: numpy.ndarray
    :param B: Response to control
    :type B: numpy.ndarray
    :param F: signal dependent noise, defaults to None
    :type F: numpy.ndarray, optional
    :param G: independent noise, defaults to None
    :type G: numpy.ndarray, optional
    :param discrete_dynamics: whether A and B are continuous or discrete, defaults to True
    :type discrete_dynamics: bool, optional
    :param noise: whether to include noise, defaults to "on"
    :type noise: str, optional
    """

    def __init__(
        self,
        timestep,
        A,
        B,
        F=None,
        G=None,
        discrete_dynamics=True,
        noise="on",
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.dim = max(A.shape)
        self.state = State()
        self.state["x"] = StateElement(
            values=numpy.zeros((self.dim, 1)),
            spaces=Space(
                [
                    -numpy.ones((self.dim, 1)) * numpy.inf,
                    numpy.ones((self.dim, 1)) * numpy.inf,
                ]
            ),
        )
        self.state_last_x = copy.copy(self.state["x"]["values"])
        self.timestep = timestep

        if F is None:
            self.F = numpy.zeros(A.shape)
        else:
            self.F = F
        if G is None:
            self.G = numpy.zeros(A.shape)
        else:
            self.G = G
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

    def finit(self):
        """finit

        Define whether to use continuous or discrete representation.
        """
        if self.bundle.user.timespace == "continuous":
            self.A = self.A_c
            self.B = self.B_c
        else:
            self.A = self.A_d
            self.B = self.B_d

    def reset(self, dic=None):
        """reset

        rorce all substates except the first to be null. Store the last state as an attribute (useful for rendering).

        .. warning::

            dic mechanism likely outdated.



        :param dic: [description], defaults to None
        :type dic: [type], optional
        """
        super().reset()
        # Force zero velocity
        self.state["x"]["values"][0][1:] = 0

        if dic is not None:
            super().reset(dic=dic)

        self.state_last_x = copy.copy(self.state["x"]["values"])

    def user_step(self, user_action):
        """user step

        Apply equations defined in the class docstring.

        .. warning ::

            call to super likely deprecated, check signature.

        :param user_action: user action
        :type user_action: `State<coopihc.space.State.State>`
        :return: (task state, task reward, is_done flag, {})
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, float, boolean, dictionnary)
        """
        # print(user_action, type(user_action))
        is_done = False
        # Call super for counters
        super().user_step(user_action)

        # For readability
        A, B, F, G = self.A, self.B, self.F, self.G
        u = user_action["values"][0]
        x = self.state["x"]["values"][0].reshape(-1, 1)

        # Generate noise samples
        if self.noise == "on":
            beta, gamma = numpy.random.normal(0, numpy.sqrt(self.timestep), (2, 1))
            omega = numpy.random.normal(0, numpy.sqrt(self.timestep), (self.dim, 1))
        else:
            beta, gamma = numpy.random.normal(0, 0, (2, 1))
            omega = numpy.random.normal(0, 0, (self.dim, 1))

        # Store last_x for render
        self.state_last_x = copy.deepcopy(self.state["x"]["values"])
        # Deterministic update + State dependent noise + independent noise
        if self.bundle.user.timespace == "discrete":
            x = (A @ x + B * u) + F @ x * beta + G @ omega
        else:
            x += (A @ x + B * u) * self.timestep + F @ x * beta + G @ omega

        self.state["x"]["values"] = x
        if abs(x[0, 0]) <= 0.01:
            is_done = True

        return self.state, 0, is_done, {}

    def assistant_step(self, assistant_action):
        """assistant_step

        Nothing.

        :param assistant_action: assistant action
        :type assistant_action: `State<coopihc.space.State.State>`
        :return: (task state, task reward, is_done flag, {})
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, float, boolean, dictionnary)
        """
        # return super().assistant_step(assistant_action)
        return self.state, 0, False, {}

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
        if (self.state_last_x[0] == self.state["x"]["values"][0]).all():
            pass
        else:
            for i in range(self.dim):
                self.axes[i].plot(
                    [
                        ((self.turn_number - 1) / 2 - 1) * self.timestep,
                        (self.turn_number - 1) / 2 * self.timestep,
                    ],
                    flatten(
                        [
                            self.state_last_x[0][i, 0].tolist(),
                            self.state["x"]["values"][0][i, 0].tolist(),
                        ]
                    ),
                    "-",
                    color=self.color[i],
                    label=self.labels[i],
                )

        return
