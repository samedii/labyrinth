import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.optim
import pyro.infer.mcmc
import matplotlib.pyplot as plt

pyro.enable_validation(True)

import game
import memory


env = game.EncodedLabyrinth()
mem = memory.DeterministicMemory()

# Gather some data through random actions
ob = env.reset()
for i in range(10):
    ac = np.random.choice(4)
    next_ob, reward, game_over = env.step(ac)

    mem.add(ob, ac, next_ob, reward, game_over)

    if game_over:
        ob = env.reset()
    else:
        ob = next_ob



