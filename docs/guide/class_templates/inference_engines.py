class MyInferenceEngine(BaseInferenceEngine):
    """ An Inference Engine that handles a Gaussian Belief. It assumes a Gaussian prior and a Gaussian likelihood. ---- Currently the covariance matrix for the likelihood is assumed to be contained by the host as self.Sigma. Maybe change this ----

    The mean and covariance matrices of Belief are stored in the substates 'MuBelief' and 'SigmaBelief'.


    :meta public:
    """
    def __init__(self, arg1):
        super().__init__()
        self.arg1 = arg1


    def infer(self):
        """
        :return: (State) state, (float) reward

        :meta public:
        """
        observation = self.observation
        # alternatively:
        # observations = self.buffer
        if self.host.role == "operator":
            state = observation['operator_state']
        else:
            state = observation["assistant_state"]

        # do something with state

        return state, 0


    def render(self, *args, **kwargs):
        mode = kwargs.get('mode')
        render_flag = False
        for r in self.render_tag:
            if r in mode:
                render_flag = True

        if 'plot' in mode:
            axtask, axoperator, axassistant = args[:3]
            if self.host.role == 'operator':
                ax = axoperator
            else:
                ax = axassistant

            if self.ax is not None:
                pass
            else:
                self.ax = ax

            # draw something

        if 'text' in mode:
            # print something
