from collections import OrderedDict
import numpy
import queue

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


# The operatormodel is not updated with this assistant
class GoalInferenceWithOperatorModelGiven(InferenceEngine):
    """ An Inference Engine used by an assistant to infer the goal of an assistant. It assumes that the operator chooses as goal one of the targets of the task, stored in the 'Targets' substate of the task.

    The inference is based on an operator_model which has to be provided to this engine.

    :param operator_model: (core.models) an operator model

    :meta public:
    """
    def __init__(self, operator_model):
        super().__init__()
        self.operator_model = operator_model

    def reset(self):
        """ Initialize the inference engine with a list of potential targets. This assumes that the task has a substate called 'Targets'

        :meta public:
        """
        self.potential_targets = self.host.bundle.game_state['task_state']['Targets']

    def generate_candidate_operator_observation(self, observation, potential_target):
        """ Generate candidate observation, i.e. treat each target as if it was the true goal. This is used to compute posteriors for each target

        :param observation: (OrderedDict) the last observation
        :param potential_target: (float) the potential target which replaces the goal state in the observation.

        :meta private:
        """
        observation['operator_state'] = OrderedDict({'Goal': [potential_target]})
        return observation


    def infer(self):
        """ Update the substate 'Beliefs' from the internal state. Generate candidate observations for each potential target, evaluate its likelihood and update the prior to form the posterior. Normalize the posterior and return the new state.

        :return: new internal state (OrderedDict), reward associated with inferring (float)

        """
        observation = self.buffer[-1]
        state = observation['assistant_state']
        old_beliefs = state['Beliefs']
        operator_action = state['OperatorAction'][0]

        for nt,t in enumerate(self.potential_targets):
            candidate_observation = self.generate_candidate_operator_observation(observation, t)
            old_beliefs[nt] *= self.operator_model.compute_likelihood(operator_action, candidate_observation)

        if sum(old_beliefs) == 0:
            print("warning: beliefs sum up to 0 after updating. I'm resetting to uniform to continue behavior. You should check if the behavior model makes sense. Here are the latest results from the model")
            old_beliefs = [1 for i in old_beliefs]
        new_beliefs = [i/sum(old_beliefs) for i in old_beliefs]
        state['Beliefs'] = new_beliefs
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
        dy = observation['task_state']['x']*self.host.timestep

        if isinstance(dy, list):
            dy = dy[0]
        if not isinstance(dy, numpy.ndarray):
            raise TypeError("Substate Xhat of {} is expected to be of type numpy.ndarray".format(type(self.host).__name__))



        if self.host.role == "operator":
            state = observation['operator_state']
            u = observation["assistant_state"]["OperatorAction"]
        else:
            state = observation['assistant_state']
            u = observation["operator_state"]["AssistantAction"]

        xhat = state["xhat"]
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
        state['xhat'] = xhat

        # Here, we use the classical definition of rewards in the LQG setup, but this requires having the true value of the state. This is not realistic --> left for future work
        # ====================== Rewards ===============

        x = self.host.bundle.task.state['x']
        if isinstance(x, list):
            x = x[0]
        if not isinstance(x, numpy.ndarray):
            raise TypeError("Substate Xhat of {} is expected to be of type numpy.ndarray".format(type(self.host).__name__))

        reward = (x-xhat).T @ self.host.U @ (x-xhat)

        return state, reward
