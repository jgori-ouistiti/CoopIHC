import numpy
import copy

from coopihc.helpers import flatten
from coopihc.space.State import State
from coopihc.space.StateElement import StateElement
from coopihc.space.Space import Space
from coopihc.interactiontask.InteractionTask import InteractionTask


class ClassicControlTask(InteractionTask):
    """verify F and G conversions."""

    def __init__(
        self,
        timestep,
        A,
        B,
        F=None,
        G=None,
        discrete_dynamics=True,
        noise="on",
    ):
        super().__init__()

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
        if self.bundle.user.timespace == "continuous":
            self.A = self.A_c
            self.B = self.B_c
        else:
            self.A = self.A_d
            self.B = self.B_d

    def reset(self, dic=None):
        super().reset()
        # Force zero velocity
        self.state["x"]["values"][0][1:] = 0

        if dic is not None:
            super().reset(dic=dic)

        self.state_last_x = copy.copy(self.state["x"]["values"])

    def user_step(self, user_action):
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
        # return super().assistant_step(assistant_action)
        return self.state, 0, False, {}

    def render(self, *args, **kwargs):
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
