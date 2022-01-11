import numpy
import copy

from coopihc.space.State import State
from coopihc.helpers import hard_flatten
from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


# The usermodel is not updated with this assistant
class GoalInferenceWithUserPolicyGiven(BaseInferenceEngine):
    """GoalInferenceWithUserPolicyGiven

    An inference Engine used by an assistant to infer the 'goal' of a user.
    The inference is based on a model of the user policy, which has to be provided to this engine.

    :param \*args: policy model
    :type \*args: :py:mod`Policy<coopihc.policy>`
    """

    def __init__(self, *args, user_policy_model=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.attach_policy(user_policy_model)
        self.render_tag = ["plot", "text"]

    def attach_policy(self, policy):
        """attach_policy

        Attach a policy to the engine from which it can sample.

        :param policy: a policy
        :type policy: :py:mod`Policy<coopihc.policy>`
        """
        if policy is None:
            self.user_policy_model = None
            return
        if not policy.explicit_likelihood:
            raise AttributeError(
                "This inference engine requires a policy defined by an explicit likelihood"
            )
        print("Attached policy {} to {}".format(policy, self.__class__.__name__))
        self.user_policy_model = policy

    def attach_set_theta(self, set_theta):
        """attach_set_theta

        The set of possible 'goal's.

        :param set_theta: dictionnary
        :type set_theta: dictionnary
        """
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
            beliefs = self.host.state["beliefs"].squeeze().tolist()

            ticks = []
            ticklabels = []
            for i, b in enumerate(beliefs):
                draw, fill, symbol = set_box(ax, 2 * i, shortcut="target", boxheight=b)
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
            beliefs = self.host.state["beliefs"].squeeze().tolist()
            print("beliefs", beliefs)

    def infer(self):
        """infer

        Update the substate 'beliefs' from the internal state. Generate candidate observations for each potential target, evaluate its likelihood and update the prior to form the posterior. Normalize the posterior and return the new state.

        :return: (new internal state, reward)
        :rtype: tuple(:py:class:`State<coopihc.space.State.State>`, float)
        """

        if self.user_policy_model is None:
            raise RuntimeError(
                "This inference engine requires a likelihood-based model of an user policy to function."
            )

        observation = self.buffer[-1]
        state = observation["assistant_state"]
        old_beliefs = state["beliefs"].squeeze().tolist()
        user_action = observation["user_action"]["action"]

        for nt, t in enumerate(self.set_theta):
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
        state["beliefs"][:] = numpy.array(new_beliefs)
        return state, 0
