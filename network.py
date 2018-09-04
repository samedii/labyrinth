
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_start = nn.Conv2d(7, 16, kernel_size=3, stride=1, padding=1)
        n_pieces = 4
        board_width = 4
        board_height = 4
        channels = 1
        self.conv_ob = nn.Conv2d(channels, n_pieces*2, kernel_size=3, stride=1, padding=1)
        self.fc_reward = nn.Linear(channels*board_width*board_height, 2*2)
        self.fc_game_over = nn.Linear(channels*board_width*board_height, 2*2)

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
        ob_logits = self.conv_ob(x).permute(0, 2, 3, 1) # categorical wants the channel last
        ob_logits_loc = ob_logits[:, :, :, :4]
        ob_logits_scale = ob_logits[:, :, :, 4:]
        x = x.view(n_obs, -1)
        reward_parms = self.fc_reward(x)
        reward_parms_loc = reward_parms[:, :2]
        reward_parms_scale = reward_parms[:, 2:]
        game_over_logits = self.fc_game_over(x)
        game_over_logits_loc = game_over_logits[:, :2]
        game_over_logits_scale = game_over_logits[:, 2:]

        return (ob_logits_loc, ob_logits_scale), (reward_parms_loc, reward_parms_scale), (game_over_logits_loc, game_over_logits_scale)

class DreamWorld(nn.Module):
    def __init__(self, cuda=False):
        super(DreamWorld, self).__init__()

        self.network = Network()

        if cuda:
            self.cuda()

    def model(self, ob, ac, next_ob, reward, game_over):
        pyro.module('dream_world', self)

        # priors
        ob_logits = pyro.sample('ob_logits', dist.Normal(
            loc=torch.zeros(ob.shape + (4,), dtype=torch.float32).cuda(),
            scale=10*torch.ones(ob.shape + (4,), dtype=torch.float32).cuda()
        ).independent())

        reward_parms = pyro.sample('reward_parms', dist.Normal(
            loc=torch.zeros(reward.shape + (2,), dtype=torch.float32).cuda(),
            scale=10*torch.ones(reward.shape + (2,), dtype=torch.float32).cuda()
        ).independent())

        game_over_logits = pyro.sample('game_over_logits', dist.Normal(
            loc=torch.zeros(game_over.shape + (2,), dtype=torch.float32).cuda(),
            scale=10*torch.ones(game_over.shape + (2,), dtype=torch.float32).cuda()
        ).independent())

        #with pyro.iarange('observe_data'): # needed?
        pyro.sample('next_ob', dist.Categorical(logits=ob_logits).independent(), obs=next_ob)
        pyro.sample('reward', dist.Normal(loc=reward_parms[:, 0], scale=F.softplus(reward_parms[:, 1])).independent(), obs=reward)
        pyro.sample('game_over', dist.Categorical(logits=game_over_logits).independent(), obs=game_over)

    def guide(self, ob, ac, next_ob, reward, game_over):
        pyro.module('dream_world', self)

        # network is only called in guide
        ob_logits, reward_parms, game_over_logits = self.network(ob, ac)

        # approximating distribution
        ob_logits = pyro.sample('ob_logits', dist.Normal(
            loc=ob_logits[0],
            scale=F.softplus(ob_logits[1])
        ).independent())

        reward_parms = pyro.sample('reward_parms', dist.Normal(
            loc=reward_parms[0],
            scale=F.softplus(reward_parms[1])
        ).independent())

        game_over_logits = pyro.sample('game_over_logits', dist.Normal(
            loc=game_over_logits[0],
            scale=F.softplus(game_over_logits[1])
        ).independent())

        return ob_logits, reward_parms, game_over_logits

    def sample(self, ob, ac):
        ob_logits, reward_parms, game_over_logits = self.guide(ob, ac, None, None, None)
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


        
