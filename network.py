
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.n_pieces = 4
        self.n_actions = 4
        self.board_height = 4
        self.board_width = 4
        self.conv_start1 = nn.Conv2d(self.n_pieces + self.n_actions, 32, kernel_size=3, stride=1, padding=1)
        self.conv_start2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)

        channels = 16
        self.conv_ob1 = nn.Conv2d(channels, 16, kernel_size=1, stride=1, padding=0)
        self.conv_ob2 = nn.Conv2d(16 + self.n_pieces, self.n_pieces*2, kernel_size=1, stride=1, padding=0)

        self.conv_reward = nn.Conv2d(channels, 8, kernel_size=1, stride=1, padding=0)
        self.fc_reward = nn.Linear(8*self.board_width*self.board_height, 2*2)

        self.conv_game_over = nn.Conv2d(channels, 8, kernel_size=1, stride=1, padding=0)
        self.fc_game_over = nn.Linear(8*self.board_width*self.board_height, 2*2)

    def forward(self, ob, ac):
        n_obs = ob.shape[0]
        ob = torch.stack([ob == x for x in range(self.n_pieces)], dim=1).float()
        ac = torch.stack([ac == x for x in range(self.n_actions)], dim=1).float()

        x = torch.cat((
            ob.view(-1, self.n_pieces, self.board_width, self.board_height),
            ac.view(-1, self.n_actions, 1, 1).repeat((1, 1, self.board_width, self.board_height))
        ), dim=1)

        x = F.relu(self.conv_start1(x))
        x = self.conv_start2(x)
        #x, _ = x.max(dim=1, keepdim=True)
        x = F.relu(x)

        ob_logits = F.relu(self.conv_ob1(x))
        ob_logits = torch.cat((ob_logits, ob), dim=1)
        ob_logits = self.conv_ob2(ob_logits)
        ob_logits = ob_logits.permute(0, 2, 3, 1) # categorical wants the channel last
        ob_logits_loc = ob_logits[:, :, :, :self.n_pieces]
        ob_logits_scale = ob_logits[:, :, :, self.n_pieces:]

        reward_logits = F.relu(self.conv_reward(x))
        reward_logits = self.fc_reward(reward_logits.view(n_obs, -1))
        reward_logits_loc = reward_logits[:, :2]
        reward_logits_scale = reward_logits[:, 2:]

        game_over_logits = F.relu(self.conv_game_over(x))
        game_over_logits = self.fc_game_over(game_over_logits.view(n_obs, -1))
        game_over_logits_loc = game_over_logits[:, :2]
        game_over_logits_scale = game_over_logits[:, 2:]

        return (ob_logits_loc, ob_logits_scale), (reward_logits_loc, reward_logits_scale), (game_over_logits_loc, game_over_logits_scale)

class CustomCategorical(dist.Categorical):
    sample_weight = 1000
    def log_prob(self, x, *args, **kwargs):
        return super().log_prob(x, *args, **kwargs)*self.sample_weight

class DreamWorld(nn.Module):
    def __init__(self, cuda=False):
        super(DreamWorld, self).__init__()

        self.network = Network()

        if cuda:
            self.cuda()

    def guide_dist(self, ob, ac):
        # network is only called in guide
        ob_logits, reward_logits, game_over_logits = self.network(ob, ac)

        # approximating distribution
        ob_logits = dist.Normal(
            loc=ob_logits[0],
            scale=F.softplus(ob_logits[1])
        ).independent()

        reward_logits = dist.Normal(
            loc=reward_logits[0],
            scale=F.softplus(reward_logits[1])
        ).independent()

        game_over_logits = dist.Normal(
            loc=game_over_logits[0],
            scale=F.softplus(game_over_logits[1])
        ).independent()

        return ob_logits, reward_logits, game_over_logits

    def model(self, ob, ac, next_ob, reward, game_over):
        pyro.module('dream_world', self)

        # priors
        ob_logits = pyro.sample('ob_logits', dist.Normal(
            loc=torch.zeros(ob.shape + (4,), dtype=torch.float32).cuda(),
            scale=10*torch.ones(ob.shape + (4,), dtype=torch.float32).cuda()
        ).independent())

        reward_logits = pyro.sample('reward_logits', dist.Normal(
            loc=torch.zeros(reward.shape + (2,), dtype=torch.float32).cuda(),
            scale=10*torch.ones(reward.shape + (2,), dtype=torch.float32).cuda()
        ).independent())

        game_over_logits = pyro.sample('game_over_logits', dist.Normal(
            loc=torch.zeros(game_over.shape + (2,), dtype=torch.float32).cuda(),
            scale=10*torch.ones(game_over.shape + (2,), dtype=torch.float32).cuda()
        ).independent())

        #with pyro.iarange('observe_data'): # needed?
        pyro.sample('next_ob', CustomCategorical(logits=ob_logits).independent(), obs=next_ob)
        pyro.sample('reward', CustomCategorical(logits=reward_logits).independent(), obs=reward)
        pyro.sample('game_over', CustomCategorical(logits=game_over_logits).independent(), obs=game_over)

    def guide(self, ob, ac, next_ob, reward, game_over):
        pyro.module('dream_world', self)

        # network is only called in guide
        ob_logits, reward_logits, game_over_logits = self.guide_dist(ob, ac)

        # approximating distribution
        ob_logits = pyro.sample('ob_logits', ob_logits)
        reward_logits = pyro.sample('reward_logits', reward_logits)
        game_over_logits = pyro.sample('game_over_logits', game_over_logits)


    def sample(self, ob, ac):
        ob_logits, reward_logits, game_over_logits = self.network(ob, ac)
        print('loc')
        print(ob_logits[0][0])
        print('scale')
        print(ob_logits[1][0])
        ob_logits, reward_logits, game_over_logits = self.guide_dist(ob, ac)
        ob_logits, reward_logits, game_over_logits = ob_logits.sample(), reward_logits.sample(), game_over_logits.sample()
        next_ob = CustomCategorical(logits=ob_logits).sample()
        reward = CustomCategorical(logits=reward_logits).sample()
        game_over = CustomCategorical(logits=game_over_logits).sample()
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


        
