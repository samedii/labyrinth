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
# mem = memory.Memory()

# Gather some data through human actions
# ob = env.reset()
# for i in range(100):
#     print(ob)
#     ac = input('action: ')
#     if ac == 'q' or ac == '':
#         break
#     ac = float(ac)
#     next_ob, reward, game_over = env.step(ac)
#     print('reward: {}'.format(reward))

#     mem.add(ob, ac, next_ob, reward, game_over)

#     if game_over:
#         print('game_over')
#         ob = env.reset()
#     else:
#         ob = next_ob

# Gather some data through random actions
ob = env.reset()
for i in range(10000):
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
        loss=pyro.infer.Trace_ELBO(num_particles=1)
    )

    ds = mem.dataset()
    print('dataset length: {}'.format(len(ds)))
    data_loader = torch.utils.data.DataLoader(ds, batch_size=5, pin_memory=cuda)

    losses = []
    for epoch in range(1, 50+1):
        print('epoch: {}'.format(epoch))
        for i, batch in enumerate(data_loader):
            loss = svi.step(*(x.cuda() for x in batch))
            losses.append(loss)

    plt.plot(losses)
    plt.show()

    # human_game.play()

    # Get uncertainty of each action
    ob, ac, next_ob, reward, game_over = (x.cuda() for x in ds[:])
    guide_dist = dream.guide_dist(ob, ac) # TODO: insert more actions, and possibly more ob based on world model?

    n_obs = len(ds)
    n_dists = 100
    kl = torch.zeros(n_obs).cuda()
    for guide in guide_dist:
        probs = dist.Categorical(logits=guide.sample((n_dists,))).probs
        index = torch.meshgrid((torch.arange(n_dists), torch.arange(n_dists)))
        probs_p = probs[index[0]]
        probs_q = probs[index[1]]
        kl_point = probs_q*torch.log(probs_q/(probs_p + 1e-6))
        kl += kl_point.sum(-1).view(n_dists*n_dists, n_obs, -1).mean(0).sum(1)
        # alternative f-divergences https://en.wikipedia.org/wiki/F-divergence
    print(kl)

    # Learn value function with kl as reward

    # Gather more data
    # todo

    # tensor([0.0151, 0.0486, 0.0162, 0.0668, 0.0550, 0.0151, 0.0235, 0.0306, 0.0304,
    #     0.0190, 0.0678, 0.0467, 0.0336, 0.0192, 0.0275, 0.0131, 0.0232, 0.0483,
    #     0.0660, 0.0136, 0.0468, 0.0319, 0.0267, 0.0225, 0.0161, 0.0086, 0.0206,
    #     0.0365, 0.0580], device='cuda:0')