
import numpy as np

class Labyrinth:

    direction = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1)
    }
    wall = 0
    empty = 1
    player = 2
    goal = 3

    def __init__(self):
        self.reset()
        self.border = self.state.shape

    def reset(self):
        self.state = np.array([
            [1, 1, 1, 1],
            [1, 0, 3, 1],
            [1, 0, 0, 0],
            [1, 1, 2, 1]
        ], dtype=np.float32)
        return self.state.copy()

    def step(self, action):
        position = np.where(self.state == self.player)
        self.state[position] = self.empty

        position = tuple(position[i] + self.direction[action][i] for i in range(2))
    
        np_position = np.array(position)

        outside_bounds = (
            (np_position <= -1).any() or
            (np_position >= self.state.shape).any()
        )

        if outside_bounds:
            return self.state, 0.0, 1.0

        win = (self.state[position] == self.goal).any()

        game_over = (
            win or
            (self.state[position] == self.wall).any()
        )

        if not game_over:
            self.state[position] = self.player

        reward = win.astype(np.float32)
        return self.state.copy(), reward, game_over.astype(np.float32)


class HumanGame:
    def __init__(self, game):
        self.game = game
    
    def play(self):
        ob = self.game.reset()

        for _ in range(100):
            print(ob)
            ac = input('action: ')
            if ac == 'q' or ac == '':
                break
            ac = float(ac)
            ob, reward, game_over = self.game.step(ac)
            print('reward: {}'.format(reward))
            if game_over:
                print('game_over')
                break
