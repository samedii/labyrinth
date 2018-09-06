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
#mem = memory.Memory()

# Gather some data through human actions
ob = env.reset()
for i in range(100):
    print(ob)
    ac = input('action: ')
    if ac == 'q' or ac == '':
        break
    ac = float(ac)
    next_ob, reward, game_over = env.step(ac)
    print('reward: {}'.format(reward))

    mem.add(ob, ac, next_ob, reward, game_over)

    if game_over:
        print('game_over')
        ob = env.reset()
    else:
        ob = next_ob

# Gather some data through random actions
ob = env.reset()
for i in range(1000):
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

class CustomLoss(pyro.infer.Trace_ELBO):
    sample_weight = 1000
    def __init__(self, num_particles=1):
        super().__init__(num_particles=num_particles)

    def loss(self, model, guide, *args, **kwargs):
        return super().loss(model, guide, *args, **kwargs)*self.sample_weight

    def loss_and_grads(self, model, guide, *args, **kwargs):
        return super().loss_and_grads(model, guide, *args, **kwargs)*self.sample_weight

pyro.clear_param_store()
for _ in range(10):

    # Learn
    svi = pyro.infer.SVI(
        dream.model,
        dream.guide,
        optim=pyro.optim.Adam({'lr': 0.005, 'betas': (0.95, 0.999)}),
        loss=CustomLoss(num_particles=10)
    )

    ds = mem.dataset()
    print('dataset length: {}'.format(len(ds)))
    data_loader = torch.utils.data.DataLoader(ds, batch_size=5, pin_memory=cuda)

    losses = []
    for epoch in range(1, 10):
        for i, (ob, ac, next_ob, reward, game_over) in enumerate(data_loader):
            ob, ac, next_ob, reward, game_over = ob.cuda(), ac.cuda(), next_ob.cuda(), reward.cuda(), game_over.cuda()
            loss = svi.step(ob, ac, next_ob, reward, game_over)
            losses.append(loss)

    plt.plot(losses)
    plt.show()

    human_game.play()

    # Get uncertainty of each action 

    # Gather more data
    # todo