from coopihc.inference.BaseInferenceEngine import BaseInferenceEngine


class ExampleInferenceEngine(BaseInferenceEngine):
    def infer(self):
        return self.state, 0


if __name__ == "__main__":
    eie = ExampleInferenceEngine()
