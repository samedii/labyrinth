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
import network

cuda = True

env = game.Labyrinth()
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

dream = network.DreamWorld(cuda)
start_ob = torch.tensor(env.reset()).view(1, 4, 4)
dream_game = network.DreamGame(dream, start_ob)
human_game = game.HumanGame(dream_game)

pyro.clear_param_store()
for _ in range(10):

    # Learn
    svi = pyro.infer.SVI(
        dream.model,
        dream.guide,
        optim=pyro.optim.Adam({'lr': 0.005, 'betas': (0.95, 0.999)}),
        loss=pyro.infer.Trace_ELBO()
    )

    data_loader = torch.utils.data.DataLoader(mem.dataset(), batch_size=5, pin_memory=cuda)

    losses = []
    for epoch in range(1, 1000):
        for i, (ob, ac, next_ob, reward, game_over) in enumerate(data_loader):
            ob, ac, next_ob, reward, game_over = ob.cuda(), ac.cuda(), next_ob.cuda(), reward.cuda(), game_over.cuda()
            loss = svi.step(ob, ac, next_ob, reward, game_over)
            losses.append(loss)

    plt.plot(losses)
    plt.show()

    human_game.play()

    # Gather more data
    # todo