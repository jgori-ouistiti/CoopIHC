from coopihc.bundle.Bundle import Bundle


class Simulator(Bundle):
    def __init__(
        self, *args, task_model=None, user_model=None, assistant=None, **kwargs
    ):
        super().__init__(
            task=task_model, user=user_model, assistant=assistant, *args, **kwargs
        )
        self.open()

    def open(self):
        self.assistant.policy._mode = "dual"
        self.assistant.inference_engine._mode = "dual"
        self.assistant.bundle = self

    def close(self):
        self.assistant.policy._mode = "primary"
        self.assistant.inference_engine._mode = "primary"
        self.assistant._simulator_close()
