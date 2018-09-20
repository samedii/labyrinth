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

    def hash(self, ob, action):
        return ( # hash("{ob}{action}")
            'ob' + str(ob) +
            'action' + str(action)
        )

    def add(self, ob, action, next_ob, reward, game_over):
        key = self.hash(ob, action)
        if key not in self.observations:
            self.observations[key] = (ob, action, next_ob, reward, game_over)

    def dataset(self):
        args = (torch.tensor(x, dtype=torch.float32) for x in zip(*self.observations.values()))
        return torch.utils.data.TensorDataset(*args)

    def get(ob, ac):
        key = self.hash(ob, action)
        return self.observations[key] if key in self.observations else None

class MemoryGameWrapper:
    def __init__(self, game, memory):
        self.game = game
        self.memory = memory

    def reset():
        self.ob = self.game.reset()
        return self.ob

    def step(ac):
        
        self.game.step(ac)

