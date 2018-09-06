import torch
import torch.utils.data

class Memory:
    observations = []

    def add(self, ob, action, next_ob, reward, game_over):
        self.observations.append((ob, action, next_ob, reward, game_over))

    def dataset(self):
        args = (torch.tensor(x, dtype=torch.float32) for x in zip(*self.observations))
        return torch.utils.data.TensorDataset(*args)

class DeterministicMemory:
    observations = {}

    def hash(self, ob, action, next_ob, reward, game_over):
        return (
            'ob' + str(ob) +
            'action' + str(action)
        )

    def add(self, ob, action, next_ob, reward, game_over):
        key = self.hash(ob, action, next_ob, reward, game_over)
        if key not in self.observations:
            self.observations[key] = (ob, action, next_ob, reward, game_over)

    def dataset(self):
        args = (torch.tensor(x, dtype=torch.float32) for x in zip(*self.observations.values()))
        return torch.utils.data.TensorDataset(*args)

# class Memory:

    # def equal_observations(a, b):
    #     for a, b in zip(a, b):
    #         if not (a == b).all():
    #             return False
    #     return True