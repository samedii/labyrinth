
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import numpy as np
import pyro.contrib.autoguide

np.random.seed(seed=999)
torch.manual_seed(999)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.n_pieces = 4
        self.n_actions = 4
        self.board_height = 4
        self.board_width = 4

        self.conv_next_ob1 = nn.Conv2d(self.n_pieces + self.n_actions - 1, 32, kernel_size=3, stride=1, padding=1)
        self.conv_next_ob2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
        self.conv_next_ob3 = nn.Conv2d(16 + self.n_pieces - 1, 16, kernel_size=1, stride=1, padding=0)
        self.conv_next_ob4 = nn.Conv2d(16, self.n_pieces, kernel_size=1, stride=1, padding=0)

        self.conv_reward1 = nn.Conv2d(self.n_pieces + self.n_actions - 1, 16, kernel_size=3, stride=1, padding=1)
        self.conv_reward2 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)
        self.conv_reward3 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.fc_reward = nn.Linear(16, 2)

        self.conv_game_over1 = nn.Conv2d(self.n_pieces + self.n_actions - 1, 16, kernel_size=3, stride=1, padding=1)
        self.conv_game_over2 = nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0)
        self.conv_game_over3 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.fc_game_over = nn.Linear(16, 2)

    def forward(self, ob, ac):
        n_obs = ob.shape[0]
        ob = torch.stack([ob == x for x in range(1, self.n_pieces)], dim=1).float()
        ac = torch.stack([ac == x for x in range(self.n_actions)], dim=1).float()

        x = torch.cat((
            ob.view(-1, self.n_pieces - 1, self.board_height, self.board_width),
            ac.view(-1, self.n_actions, 1, 1).repeat((1, 1, self.board_height, self.board_width))
        ), dim=1)

        next_ob_logits = F.relu(self.conv_next_ob1(x))
        next_ob_logits = F.relu(self.conv_next_ob2(next_ob_logits))
        next_ob_logits = torch.cat((next_ob_logits, ob), dim=1)
        next_ob_logits = F.relu(self.conv_next_ob3(next_ob_logits))
        next_ob_logits = self.conv_next_ob4(next_ob_logits)
        next_ob_logits = next_ob_logits.permute(0, 2, 3, 1) # categorical wants the channel last

        reward_logits = F.relu(self.conv_reward1(x))
        reward_logits = F.relu(self.conv_reward2(reward_logits))
        reward_logits = self.conv_reward3(reward_logits)
        reward_logits = reward_logits.view(n_obs, -1)
        reward_logits, _ = reward_logits.sort()
        reward_logits = self.fc_reward(reward_logits)

        game_over_logits = F.relu(self.conv_game_over1(x))
        game_over_logits = F.relu(self.conv_game_over2(game_over_logits))
        game_over_logits = self.conv_game_over3(game_over_logits)

        # 331
        # game_over_logits, _ = game_over_logits[:, 1].view(n_obs, -1).max(dim=-1)
        # game_over_logits = torch.stack((torch.zeros_like(game_over_logits), game_over_logits), dim=1)

        # 0.1
        game_over_logits = game_over_logits.view(n_obs, -1)
        game_over_logits, _ = game_over_logits.sort()
        game_over_logits = self.fc_game_over(game_over_logits)

        # 277
        # game_over_logits, _ = game_over_logits.view(n_obs, 2, -1).max(dim=-1)

        # game_over_logits = game_over_logits.view(n_obs, 2, -1)
        # game_over_logits = game_over_logits*F.log_softmax(game_over_logits[:, [1]], dim=-1)
        # game_over_logits = game_over_logits.sum(-1)

        # game_over_logits = F.log_softmax(game_over_logits, dim=-3)
        # game_over_logits = game_over_logits[:, 0].sum(dim=(-2, -1))
        # game_over_logits = game_over_logits - torch.log(1 - torch.exp(game_over_logits))
        # game_over_logits = torch.stack((game_over_logits, torch.zeros_like(game_over_logits)), dim=1)

        # game_over_logits = game_over_logits.view(n_obs, 2, -1)
        # game_over_index = game_over_logits[:, 1].argmax(dim=-1)
        # game_over_logits = game_over_logits[np.arange(n_obs), :, game_over_index]

        return next_ob_logits, reward_logits, game_over_logits

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

        self.names_and_sizes = [
            (name[len('network.'):], tensor.shape)
            for name, tensor in self.named_parameters()
            if name in [
                'network.conv_next_ob3.weight',
                'network.conv_next_ob3.bias',
                'network.conv_reward3.weight',
                'network.conv_reward3.bias',
                # 'network.conv_game_over3.weight',
                # 'network.conv_game_over3.bias',
            ]
        ]

        print(self.names_and_sizes)

        self.prior_dist = {
                name: dist.Normal(
                    loc=torch.zeros(size).cuda(),
                    scale=10.0*torch.ones(size).cuda()
                ).independent()
                for name, size in self.names_and_sizes
            }

    def model(self, ob, ac, next_ob, reward, game_over):
        network_dist = pyro.random_module('dream_world', self.network, self.prior_dist)
        network = network_dist()
        next_ob_logits, reward_logits, game_over_logits = network(ob, ac)

        next_ob_batch = pyro.iarange('next_ob_batch', next_ob.shape[0], dim=-3)
        next_ob_height = pyro.iarange('next_ob_height', next_ob.shape[1], dim=-2)
        next_ob_width = pyro.iarange('next_ob_width', next_ob.shape[2], dim=-1)
        with next_ob_batch, next_ob_height, next_ob_width:
            next_ob = pyro.sample('next_ob', CustomCategorical(logits=next_ob_logits), obs=next_ob)

        with pyro.iarange('batch', len(ac)):
            reward = pyro.sample('reward', CustomCategorical(logits=reward_logits), obs=reward)
            game_over = pyro.sample('game_over', CustomCategorical(logits=game_over_logits), obs=game_over)

    def guide(self, ob, ac, next_ob, reward, game_over):
        guide_dist = {
            name: dist.Normal(
                loc=pyro.param('{}.{}'.format(name, 'loc'), 0.01*torch.randn(size).cuda()),
                scale=F.softplus(pyro.param('{}.{}'.format(name, 'scale'), 0.01*torch.randn(size).cuda()))
            ).independent()
            for name, size in self.names_and_sizes
        }
        network_dist = pyro.random_module('dream_world', self.network, guide_dist)
        return network_dist()

    #def forward(self, ob, ac):
    #    return self.sample(ob, ac)

    def sample(self, ob, ac):
        network_sample = self.guide(ob, ac, None, None, None)
        next_ob_logits, reward_logits, game_over_logits = network_sample(ob, ac)
        next_ob = CustomCategorical(logits=next_ob_logits).sample()
        reward = CustomCategorical(logits=reward_logits).sample()
        game_over = CustomCategorical(logits=game_over_logits).sample()
        return next_ob, reward, game_over

    def model_uncertainty(self, ob, ac, n_samples=100):
        # alternative f-divergences https://en.wikipedia.org/wiki/F-divergence
        n_obs = ob.shape[0]
        network_samples = [self.guide(ob, ac, None, None, None) for _ in range(n_samples)]
        network_outputs = [x(ob, ac) for x in network_samples]

        hellinger = []
        for logits in zip(*network_outputs): # next_ob, reward, game_over
            logits = torch.stack(logits, dim=0)
            probs = dist.Categorical(logits=logits).probs
            index = torch.meshgrid((torch.arange(n_samples), torch.arange(n_samples)))
            probs_p = probs[index[0]]
            probs_q = probs[index[1]]
            #kl_point = probs_q*torch.log(probs_q/probs_p)
            #kl_point[~torch.isfinite(kl_point)] = 100 # ugly hack, set to something large?
            #kl += kl_point.sum(-1).view(n_samples*n_samples, n_obs, -1).mean(0).sum(1)
            hellinger_point = (torch.sqrt(probs_p) - torch.sqrt(probs_q))**2
            hellinger.append(torch.sqrt(hellinger_point.sum(-1)/2).view(n_samples*n_samples, n_obs, -1).mean(0).mean(1))
        return torch.stack(hellinger, dim=1).mean(1) #*torch.tensor([0.6, 0.2, 0.2]).cuda()


class DreamGame():
    def __init__(self, dream, ob, memory=None):
        self.start_ob = torch.tensor(ob).view(1, 4, 4).cuda() # hack
        self.dream = dream
        self.memory = memory

    def reset(self):
        self.ob = self.start_ob
        return self.ob

    def step(self, ac):
        ac = torch.tensor([ac]).cuda() # hack
        self.ob, reward, game_over = self.dream.sample(self.ob, ac)
        return self.ob[0], reward[0], game_over[0]

        
