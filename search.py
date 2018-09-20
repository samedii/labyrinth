
import numpy as np
import torch

class QLearning:
    def __init__(self, dream, ob, ac, next_ob, game_over, n_samples=10):
        n_actions = 4

        # Q-learning of model uncertainty
        unique_ob = torch.stack(tuple({str(x): x for x in ob}.values()), dim=0) # remove duplicates
        ob_lookup = {str(x): index for index, x in enumerate(unique_ob)}

        # Get uncertainty of each action and connections
        # Could repeat this step to handle uncertainty of transitions better
        next_ob = next_ob.view(-1, n_actions, next_ob.shape[1], next_ob.shape[2])
        game_over = game_over.view(-1, 4).byte()
        kl = dream.model_uncertainty(ob, ac, n_samples=n_samples).view(-1, n_actions)
        index = -torch.ones((len(ob_lookup), 4)).cuda()
        for i, x in enumerate(unique_ob):
            # there is probably a better way of doing this
            point_index = (next_ob == x.long()).all(-1).all(-1)
            index[point_index] = torch.tensor(i).float().cuda()
        index[game_over] = 0
        index = torch.cat((index, -torch.ones((1, 4)).cuda()), dim=0).long()
        kl = torch.cat((kl, torch.zeros((1, 4)).cuda()), dim=0)
        print('kl', kl)
        print('index', index)

        # Propagate uncertainty values backwards to get the discounted sum
        learning_rate = 0.45
        discount_factor = 0.99
        value_kl = kl
        for _ in range(1000):
            connection_value = value_kl[index]
            max_connection_value, _ = connection_value.max(dim=2)
            value_kl = (1 - learning_rate)*value_kl + learning_rate*(kl + discount_factor*max_connection_value)
        print('value_kl', value_kl)

        self.kl = kl
        self.value_kl = value_kl
        self.ob_lookup = ob_lookup
        self.unique_ob = unique_ob

    def get_action(self, ob):
        key = str(torch.from_numpy(ob).cuda())
        if key in self.ob_lookup:
            i = self.ob_lookup[key]
            ac = self.value_kl[i].argmax().item()
            print('ac', ac, 'value_kl[i]', self.value_kl[i])
        else:
            ac = np.random.choice(4)
            print('random action, ac', ac)
        return ac

    def directions(self):
        board = np.nan*torch.ones((4, 4)).float().cuda()
        board_value_kl = np.nan*torch.ones((4, 4)).float().cuda()
        board_kl = np.nan*torch.ones((4, 4)).float().cuda()
        for ob in self.unique_ob:
            key = str(ob)
            i = self.ob_lookup[key]
            ac = self.value_kl[i].argmax()
            value_kl = self.value_kl[i].max()
            kl = self.kl[i].max()

            # if not np.isnan(board[ob == 2].cpu().numpy()).all():
            #     print('AGAIN')
            #     print(torch.nonzero(ob == 2))
            #     print(ac)
            if np.isnan(board[ob == 2].cpu().numpy()).all() and (ob == 2).sum() == 1:
                board[ob == 2] = ac.float()
                board_value_kl[ob == 2] = value_kl
                board_kl[ob == 2] = kl
        return board, board_value_kl, board_kl

