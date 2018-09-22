import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.optim
import pyro.contrib.autoguide
import matplotlib.pyplot as plt

pyro.enable_validation(True)
#pyro.clear_param_store()

import game
import search
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
for i in range(100):
    ac = np.random.choice(4)
    next_ob, reward, game_over = env.step(ac)

    mem.add(ob, ac, next_ob, reward, game_over)

    if game_over:
        ob = env.reset()
    else:
        ob = next_ob

dream = network.DreamWorld(cuda)
dream.cuda()
start_ob = torch.tensor(env.reset()).view(1, 4, 4)
dream_game = network.DreamGame(dream, start_ob)
human_game = game.HumanGame(dream_game)


for _ in range(100):

    # Learn
    svi = pyro.infer.SVI(
        dream.model,
        dream.guide,
        optim=pyro.optim.Adam({'lr': 0.002, 'betas': (0.95, 0.999)}),
        loss=pyro.infer.TraceGraph_ELBO(num_particles=1)
    )

    ds = mem.dataset()
    print('dataset length: {}'.format(len(ds)))
    data_loader = torch.utils.data.DataLoader(ds, batch_size=5, pin_memory=cuda) #, shuffle=True)

    import time

    losses = []
    for epoch in range(1, 100+1):
        # start = time.time()
        for i, batch in enumerate(data_loader):
            start = time.time()
            loss = svi.step(*(x.cuda() for x in batch))
            losses.append(loss)
        # print('epoch: {}, loss: {}'.format(epoch, loss))
        # print(time.time() - start)
    print('loss: {}'.format(loss))
    # plt.plot(losses)
    # plt.show()

    # human_game.play()

    # Simulate new moves
    n_dream_moves = 1
    ob, _, next_ob, _, game_over = (x.cuda() for x in ds[:])
    n_actions = 4
    for _ in range(n_dream_moves):
        next_ob = next_ob[game_over == 0]
        ob = torch.cat((ob, next_ob.float()), dim=0)
        ob = torch.stack(tuple({str(x): x for x in ob}.values()), dim=0) # remove duplicates

        ac = torch.arange(n_actions).float().cuda()
        ac = ac.view(1, n_actions).repeat(ob.shape[0], 1).view(-1)
        ob = ob.view(-1, 1, 4, 4).repeat(1, n_actions, 1, 1).view(-1, 4, 4)

        next_ob, _, game_over = dream.sample(ob, ac)

        # optimization: save in a dict and do not recalculate stuff

    q_learning = search.QLearning(dream, ob, ac, next_ob, game_over, n_samples=50)

    board, board_value_uncertainty, board_uncertainty = q_learning.directions()
    print('board')
    print(board.cpu().numpy())
    print(board_value_uncertainty)
    print(board_uncertainty)

    # Alternative search methods:
    #   Breadth first search (can drop game overs with less value)
    #   Depth first monte carlo (ineffective? Zero value of going back someplace)
    #   Optimization (needs relaxation of categorical?)

    # Gather data
    ob = env.reset()
    previous_ob = ob
    for _ in range(20):

        print('ob', ob)
        ac = q_learning.get_action(ob)

        next_ob, reward, game_over = env.step(ac)
        mem.add(ob, ac, next_ob, reward, game_over)

        if reward > 0:
            raise Exception('Found a reward!')

        if game_over:
            print('Game over')
            break
        elif (previous_ob == next_ob).all():
            print('Returned to previous state')
            break
        else:
            previous_ob = ob
            ob = next_ob