import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.optim
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

    # plt.plot(losses)
    # plt.show()

    # human_game.play()

    # Simulate new moves
    ob, _, next_ob, _, game_over = (x.cuda() for x in ds[:])
    n_actions = 4
    for _ in range(10):
        next_ob = next_ob[game_over == 0]
        ob = torch.cat((ob, next_ob.float()), dim=0)
        ob = torch.stack(tuple({str(x): x for x in ob}.values()), dim=0) # remove duplicates

        ac = torch.arange(n_actions).float().cuda()
        ac = ac.view(1, n_actions).repeat(ob.shape[0], 1).view(-1)
        ob = ob.view(-1, 1, 4, 4).repeat(1, n_actions, 1, 1).view(-1, 4, 4)

        next_ob, _, game_over = dream.sample(ob, ac)

        # optimization: save in a dict and do not recalculate stuff

    # Q-learning of model uncertainty
    unique_ob = torch.stack(tuple({str(x): x for x in ob}.values()), dim=0) # remove duplicates
    ob_lookup = {str(x): index for index, x in enumerate(unique_ob)}

    # Get uncertainty of each action and connections
    # Could repeat this step to handle uncertainty of transitions better
    kl = dream.model_uncertainty(ob, ac, n_samples=10).view(-1, n_actions)
    index = -torch.ones((len(ob_lookup), 4)).cuda()
    point_index = [(next_ob == x.long()).all(-1).all(-1).view(-1, n_actions) for x in unique_ob]
    game_over = game_over
    for i, x in enumerate(point_index):
        # there is probably a better way of doing this
        # TODO: check index, it might be wrong?
        index[x] = torch.where(game_over.view(-1, 4)[x] == 1, torch.tensor(-1.0).cuda(), torch.tensor(i).float().cuda())
        kl[x] = torch.where(game_over.view(-1, 4)[x] == 1, torch.tensor(0.0).cuda(), kl[x])
    index = torch.cat((index, -torch.ones((1, 4)).cuda()), dim=0).long()
    kl = torch.cat((kl, torch.zeros((1, 4)).cuda()), dim=0)
    print('kl', kl)
    print('index', index)

    # Propagate uncertainty values backwards to get the discounted sum
    learning_rate = 0.5
    discount_factor = 0.9
    value_kl = kl
    for _ in range(100):
        connection_value = value_kl[index]
        max_connection_value, _ = connection_value.max(dim=1)
        value_kl = (1 - learning_rate)*value_kl + learning_rate*(kl + discount_factor*max_connection_value)
    print('value_kl', value_kl)

    # Alternative search methods:
    #   Breadth first search
    #   Depth first monte carlo (ineffective?)
    #   Optimization (needs relaxation of categorical?)

    # TODO:
    # Vectorized dream world with uncertainty as a reward
    # Monte carlo that shit
    # Follow best monte carlo to get more data

    # Learn value function with kl as reward

    # Find N actions that give the highest uncertainty just through backprop?

    # Gather more data
    # todo
