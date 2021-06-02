from collections import OrderedDict
import numpy
import queue
from core.space import State
from core.helpers import hard_flatten
import copy


# Base Inference Engine: does nothing but return the same state. Any new inference method can subclass InferenceEngine to have a buffer and add_observation method (required by the bundle)







class InferenceEngine:
    """ An Inference Engine. Subclass this to define new Inference Engines.

    An inference_engine provides a buffer, whose depth is defined during initializing. The buffer automatically stores the newest observation and pushes back the oldest one.

    The infer API method has to be redefined when subclassing this class.

    :param buffer_depth: \[optional\] (int) size of the buffer used to store observations. a size 0 can be used in which case the observation is stored in inference_engine.observation.
    :init_values: \[optional\] (float) values to which the buffer should be initialized

    :meta public:
    """
    def __init__(self, buffer_depth = 1):
        self.buffer = None
        self.buffer_depth = buffer_depth
        self.render_flag = None

    def add_observation(self, observation):
        """ add an observation  to the buffer. Currently poorly performing implementation

        :param observation: verify type.
        """

        if self.buffer is None:
            self.buffer = []
        if len(self.buffer) < self.buffer_depth:
            self.buffer.append(observation)
        else:
            self.buffer = self.buffer[1:] + [observation]



    def infer(self):
        """ The main method of this class.

        Return the new value of the internal state of the agent, as well as the reward associated with inferring the . By default, this inference engine does nothing, and just returns the state.

        :return: new_internal_state (OrderedDict), reward (float)
        """
        # do something with information inside buffer

        if self.host.role == "operator":
            try:
                return self.buffer[-1]['operator_state'], 0
            except KeyError:
                return OrderedDict({}), 0
        elif self.host.role == "assistant":
            try:
                return self.buffer[-1]['assistant_state'], 0
            except KeyError:
                return OrderedDict({}), 0

    def render(self, *args, **kwargs):
        mode = kwargs.get('mode')

        render_flag = False
        for r in self.render_tag:
            if r in mode:
                render_flag = True

        if render_flag:

            if 'plot' in mode:
                ax = args[0]
                if self.ax is not None:
                    pass
                else:
                    self.ax = ax
                    self.ax.set_title(type(self).__name__)

            if 'text' in mode:
                print(type(self).__name__)



# The operatormodel is not updated with this assistant
class GoalInferenceWithOperatorPolicyGiven(InferenceEngine):
    """ An Inference Engine used by an assistant to infer the goal of an operator.

    The inference is based on an operator_model which has to be provided to this engine.

    :meta public:
    """
    def __init__(self, *args):
        super().__init__()
        try:
            self.attach_policy(args[0])
        except IndexError:
            self.operator_policy_model = None

        self.render_tag = ['plot', 'text']
        self.ax = None


    def attach_policy(self, policy):
        if not policy.explicit_likelihood:
            print('Warning: This inference engine requires a policy defined by an explicit likelihood')
        print('Attached policy {} to {}'.format(policy, self.__class__.__name__))
        self.operator_policy_model = policy

    def attach_set_theta(self, set_theta):
        self.set_theta = set_theta

    def render(self, *args, **kwargs):

        mode = kwargs.get('mode')

        render_flag = False
        for r in self.render_tag:
            if r in mode:
                render_flag = True

        ## ----------------------------- Begin Helper functions
        def set_box(ax, pos, draw = "k", fill = None, symbol = None, symbol_color = None, shortcut = None, box_width = 1, boxheight = 1, boxbottom = 0):
            if shortcut == 'void':
                draw = 'k'
                fill = '#aaaaaa'
                symbol = None
            elif shortcut == 'target':
                draw = '#96006c'
                fill = "#913979"
                symbol = "1"
                symbol_color = 'k'
            elif shortcut == 'goal':
                draw = '#009c08'
                fill = '#349439'
                symbol = "X"
                symbol_color = 'k'
            elif shortcut == 'position':
                draw = '#00189c'
                fill = "#6573bf"
                symbol = "X"
                symbol_color = 'k'

            BOX_HW = box_width/2
            _x = [pos-BOX_HW, pos+BOX_HW, pos + BOX_HW, pos - BOX_HW]
            _y = [boxbottom, boxbottom, boxbottom + boxheight, boxbottom + boxheight]
            x_cycle = _x + [_x[0]]
            y_cycle = _y + [_y[0]]
            if fill is not None:
                fill = ax.fill_between(_x[:2], _y[:2], _y[2:], color = fill)

            draw, = ax.plot(x_cycle,y_cycle, '-', color = draw, lw = 2)
            symbol = None
            if symbol is not None:
                symbol = ax.plot(pos, 0, color = symbol_color, marker = symbol, markersize = 100)

            return draw, fill, symbol

        def draw_beliefs(ax):
            beliefs = hard_flatten(self.host.state['Beliefs']['human_values'])
            ticks = []
            ticklabels = []
            for i, b in enumerate(beliefs):
                draw, fill, symbol = set_box(ax, 2*i, shortcut = 'target', boxheight = b)
                ticks.append(2*i)
                ticklabels.append(i)
            self.ax.set_xticks(ticks)
            self.ax.set_xticklabels(ticklabels, rotation = 90)

    ## -------------------------- End Helper functions

        if 'plot' in mode:
            ax = args[0]
            if self.ax is not None:
                title = self.ax.get_title()
                self.ax.clear()
                draw_beliefs(ax)
                ax.set_title(title)

            else:
                self.ax = ax
                draw_beliefs(ax)
                self.ax.set_title(type(self).__name__ + " Beliefs")

        if 'text' in mode:
            beliefs = hard_flatten(self.host.state['Beliefs']['human_values'])
            print("Beliefs", beliefs)

    def infer(self):
        """ Update the substate 'Beliefs' from the internal state. Generate candidate observations for each potential target, evaluate its likelihood and update the prior to form the posterior. Normalize the posterior and return the new state.

        :return: new internal state (OrderedDict), reward associated with inferring (float)

        """

        if self.operator_policy_model is None:
            raise RuntimeError('This inference engine requires a likelihood-based model of an operator policy to function.')


        observation = self.buffer[-1]
        state = observation['assistant_state']
        old_beliefs = state['Beliefs']['values']
        operator_action = observation['operator_action']['action']

        for nt,t in enumerate(self.set_theta):
            candidate_observation = copy.deepcopy(observation)
            for key, value in t.items():
                try:
                    candidate_observation[key[0]][key[1]] = value
                except KeyError: # key[0] is not in observation
                    _state = State()
                    _state[key[1]] = value
                    candidate_observation[key[0]] = _state

            old_beliefs[nt] *= self.operator_policy_model.compute_likelihood(operator_action, candidate_observation)


        if sum(old_beliefs) == 0:
            print("warning: beliefs sum up to 0 after updating. I'm resetting to uniform to continue behavior. You should check if the behavior model makes sense. Here are the latest results from the model")
            old_beliefs = [1 for i in old_beliefs]
        new_beliefs = [i/sum(old_beliefs) for i in old_beliefs]
        print(new_beliefs)
        state['Beliefs']['values'] = new_beliefs
        return state, 0








class ContinuousGaussian(InferenceEngine):
    """ An Inference Engine that handles a Gaussian Belief. It assumes a Gaussian prior and a Gaussian likelihood. ---- Currently the covariance matrix for the likelihood is assumed to be contained by the host as self.Sigma. Maybe change this ----

    The mean and covariance matrices of Belief are stored in the substates 'MuBelief' and 'SigmaBelief'.

    .. warning::
        This inference engine requires a function yms to be attached via attach_yms() which specifies which component of the observation to use as observation for the updating part i.e. which substate is modeled by the belief.

    :meta public:
    """
    def __init__(self):
        super().__init__()
        self.yms = None

    def reset(self, *args):
        # Check if needed, if so add it to the Base class, otherwise remove
        """
        :meta private:
        """
        pass


    def attach_yms(self, yms):
        """ Call this when initializing the inference engine.

        yms is a function which takes an observation (OrderedDict) as input, and returns the substate that is modeled by the belief. See the example below where a target is modeled.

        .. code-block:: python

            def yms(internal_observation):
                ## specify here what part of the internal observation will be used as observation sample
                return internal_observation['task_state']['Targets'][0]

        :param yms: (method) see above.

        """
        self.yms = yms


    def infer(self):
        """ Update the Gaussian Beliefs, see XX for more information.

        :return: (OrderedDict) state, (float) 0

        :meta public:
        """
        observation = self.buffer[-1]
        if self.host.role == "operator":
            state = observation['operator_state']
        else:
            state = observation["assistant_state"]

        if self.yms is None:
            print("Please call attach_yms() method before. You have to specify which components of the states constitute the observation that is used to update the beliefs.")
        else:
            y = numpy.array(self.yms(observation))

        oldmu, oldsigma = state['MuBelief'], state['SigmaBelief']
        new_sigma = numpy.linalg.inv((numpy.linalg.inv(oldsigma) + numpy.linalg.inv(self.host.Sigma)))
        newmu = new_sigma @ (numpy.linalg.inv(self.host.Sigma)@y + numpy.linalg.inv(oldsigma)@oldmu)
        state['MuBelief'] = newmu
        state['SigmaBelief'] = new_sigma
        return state, 0


class ContinuousKalmanUpdate(InferenceEngine):
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
            raise RuntimeError('You have to set the forward model dynamics, by calling the set_forward_model_dynamics() method with inference engine {}  before using it'.format(type(self).__name__))
        if not self.K_flag:
            raise RuntimeError('You have to set the K Matrix, by calling the set_K() method with inference engine {}  before using it'.format(type(self).__name__))
        observation = self.buffer[-1]
        dy = observation['task_state']['_value_x']*self.host.timestep

        if isinstance(dy, list):
            dy = dy[0]
        if not isinstance(dy, numpy.ndarray):
            raise TypeError("Substate Xhat of {} is expected to be of type numpy.ndarray".format(type(self.host).__name__))


        state = observation['{}_state'.format(self.host.role)]
        u = observation['{}_action'.format(self.host.role)]['_value_action']


        xhat = state["_value_xhat"]
        if isinstance(xhat, list):
            xhat = xhat[0]
        if not isinstance(xhat, numpy.ndarray):
            raise TypeError("Substate Xhat of {} is expected to be of type numpy.ndarray".format(type(self.host).__name__))

        if isinstance(u, list):
            u = u[0]
        if not isinstance(u, numpy.ndarray):
            raise TypeError("Substate Xhat of {} is expected to be of type numpy.ndarray".format(type(self.host).__name__))

        xhat = xhat.reshape(-1,1)
        u = u.reshape(-1,1)
        deltaxhat = (self.A @ xhat + self.B @ u)*self.host.timestep + self.K @ (dy - self.C @ xhat * self.host.timestep)
        xhat += deltaxhat
        state['_value_xhat'] = xhat

        # Here, we use the classical definition of rewards in the LQG setup, but this requires having the true value of the state. This may or may not realistic...
        # ====================== Rewards ===============

        x = self.host.bundle.task.state['_value_x']
        if isinstance(x, list):
            x = x[0]
        if not isinstance(x, numpy.ndarray):
            raise TypeError("Substate Xhat of {} is expected to be of type numpy.ndarray".format(type(self.host).__name__))

        reward = (x-xhat).T @ self.host.U @ (x-xhat)

        return state, reward
