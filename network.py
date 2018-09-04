
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_start = nn.Conv2d(7, 16, kernel_size=3, stride=1, padding=1)
        channels = 1
        self.conv_ob = nn.Conv2d(channels, 4, kernel_size=3, stride=1, padding=1)
        self.fc_reward = nn.Linear(channels*4*4, 2)
        self.fc_game_over = nn.Linear(channels*4*4, 2)

    def forward(self, ob, ac):
        n_obs = ob.shape[0]
        ob = torch.stack([ob == x for x in range(1, 4)], dim=1).float()
        ac = torch.stack([ac == x for x in range(4)], dim=1).float()

        x = torch.cat((
            ob.view(-1, 3, 4, 4),
            ac.view(-1, 4, 1, 1).repeat((1, 1, 4, 4))
        ), dim=1)

        x = self.conv_start(x)
        x, _ = x.max(dim=1, keepdim=True)
        x = F.relu(x)
        ob_logits = self.conv_ob(x)
        x = x.view(n_obs, -1)
        reward_parms = self.fc_reward(x)
        game_over_logits = self.fc_game_over(x)

        return ob_logits, reward_parms, game_over_logits

class DreamWorld(nn.Module):
    def __init__(self, cuda=False):
        super(DreamWorld, self).__init__()

        self.network = Network()

        if cuda:
            self.cuda()

    def model(self, ob, ac, next_ob, reward, game_over):

        pyro.module('network', self.network)

        ob_logits, reward_parms, game_over_logits = self.network(ob, ac)
        #with pyro.iarange('observe_data'): # needed?
        pyro.sample('next_ob', dist.Categorical(logits=ob_logits).independent(), obs=next_ob)
        pyro.sample('reward', dist.Normal(loc=reward_parms[:, 0], scale=F.softplus(reward_parms[:, 1])).independent(), obs=reward)
        pyro.sample('game_over', dist.Categorical(logits=game_over_logits).independent(), obs=game_over)

    def guide(self, ob, ac, next_ob, reward, game_over):
        pass

    def sample(self, ob, ac):
        ob_logits, reward_parms, game_over_logits = self.network(ob, ac)
        next_ob = dist.Categorical(logits=ob_logits).sample()
        reward = dist.Normal(loc=reward_parms[:, 0], scale=F.softplus(reward_parms[:, 1])).sample()
        game_over = dist.Categorical(logits=game_over_logits).sample()
        return next_ob, reward, game_over


class DreamGame():
    def __init__(self, dream, ob):
        self.start_ob = torch.tensor(ob).view(1, 4, 4).cuda() # hack
        self.dream = dream

    def reset(self):
        self.ob = self.start_ob
        return self.ob

    def step(self, ac):
        ac = torch.tensor([ac]).cuda() # hack
        self.ob, reward, game_over = self.dream.sample(self.ob, ac)
        return self.ob[0], reward[0], game_over[0]


        
