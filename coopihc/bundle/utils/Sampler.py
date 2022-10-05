class Sampler:
    def __init__(self, bundle, n_turns=100):
        self.bundle = bundle
        self.n_turns = n_turns

    def __iter__(self):

        self.iter_idx = 0
        return self

    def __next__(self):
        self.iter_idx += 1
        if self.iter_idx <= self.n_turns:
            init_state = self.bundle.reset()
            states, rewards = [init_state], [None]
            while True:
                state, reward, is_done = self.bundle.step()
                states.append(state)
                rewards.append(reward)
                if is_done:
                    break
            return (states, rewards)

        else:
            raise StopIteration
