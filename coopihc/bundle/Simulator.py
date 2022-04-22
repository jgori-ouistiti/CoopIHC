from coopihc.bundle.Bundle import Bundle


class Simulator(Bundle):
    def __init__(
        self,
        name="Simulator",
        task_model=None,
        user_model=None,
        assistant=None,
        use_primary_inference=False,
        use_primary_policy=False,
        **kwargs
    ):
        self.name = name

        super().__init__(
            task=task_model, user=user_model, assistant=assistant, **kwargs
        )

        self.use_primary_inference = use_primary_inference
        self.use_primary_policy = use_primary_policy

    def open(self):
        if not self.use_primary_policy:
            self.assistant.policy._mode = "dual"
        if not self.use_primary_inference:
            self.assistant.inference_engine._mode = "dual"
        self.assistant.bundle = self

    def close(self):
        self.assistant.policy._mode = "primary"
        self.assistant.inference_engine._mode = "primary"
        self.assistant._simulator_close()
