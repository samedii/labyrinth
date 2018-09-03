
import numpy as np

class Labyrinth:

    direction = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1)
    }

    def __init__(self):
        self.reset()
        self.border = self.state.shape

    def reset(self):
        self.state = np.array([
            [0, 0, 0, 0],
            [0, 2, 3, 0],
            [0, 2, 2, 2],
            [0, 0, 1, 0]
        ], dtype=np.float32)
        return self.state

    def step(self, action):
        position = np.where(self.state == 1)
        self.state[position] = 0

        position = tuple(position[i] + self.direction[action][i] for i in range(2))
    
        np_position = np.array(position)
        game_over = (
            (np_position <= -1).any() or
            (np_position >= self.state.shape).any() or
            (self.state[position] == 2).any()
        ).astype(np.float32)

        if not game_over:
            self.state[position] = 1
        
        reward = (self.state != 3).all().astype(np.float32) # player standing on goal

        return self.state, reward, game_over

class EncodedLabyrinth:

    def __init__(self):
        self.labyrinth = Labyrinth()

    def encode(self, state):
        return np.array([state == x for x in range(4)], dtype=np.float32)

    def reset(self):
        state = self.labyrinth.reset()
        return self.encode(state)

    def step(self, action):
        state, reward, game_over = self.labyrinth.step(action)
        return self.encode(state), reward, game_over

