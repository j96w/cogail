import numpy as np
import torch
import torch.nn as nn

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, recode_dim, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        self.recode = nn.Sequential(init_(nn.Linear(recode_dim, 64)), nn.ReLU(),
                                    init_(nn.Linear(64, 2)), nn.Tanh())

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.half_action_space = int(num_outputs / 2)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, random_seeds, rnn_hxs, masks, deterministic=False):
        # print(inputs)
        value, actor_features, rnn_hxs = self.base(inputs, random_seeds, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()


        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, random_seeds, rnn_hxs, masks):
        value, _, _ = self.base(inputs, random_seeds, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, random_seeds, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, random_seeds, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        pred_action = dist.mode()

        input_code = torch.cat((inputs, pred_action[:, :self.half_action_space]), dim=1)
        pred_code = self.recode(input_code)

        return value, action_log_probs, dist_entropy, rnn_hxs, pred_code

    def evaluate_code(self, inputs, action):
        input_code = torch.cat((inputs, action[:, :self.half_action_space]), dim=1)
        pred_code = self.recode(input_code)

        return pred_code

    def process_exp_dataset(self, inputs, action, ids):
        bs = inputs.size()[0]
        if bs > 128:
            max_size = 64
        else:
            max_size = int(bs / 2)

        input_code = torch.cat((inputs, action[:, :self.half_action_space]), dim=1)
        pred_code = self.recode(input_code)

        value, actor_features, _ = self.base(inputs, pred_code, None, None)
        dist = self.dist(actor_features)

        pred_action = dist.mode()

        distance = torch.norm((pred_action - action), dim=1).view(-1)
        data_sorted, data_index = torch.sort(distance, dim=0, descending=True)

        old_inputs = inputs[:max_size]
        old_actions = action[:max_size]
        old_codes = pred_code[:max_size]
        old_ids = ids[:max_size]

        data_index = data_index.view(-1)[:max_size]

        new_inputs = torch.index_select(inputs, 0, data_index)
        new_actions = torch.index_select(action, 0, data_index)
        new_codes = torch.index_select(pred_code, 0, data_index).detach()
        new_ids = torch.index_select(ids, 0, data_index)

        final_inputs = torch.cat((old_inputs, new_inputs), dim=0)
        final_actions = torch.cat((old_actions, new_actions), dim=0)
        final_codes = torch.cat((old_codes, new_codes), dim=0).detach()
        final_ids = torch.cat((old_ids, new_ids), dim=0)

        return final_inputs, final_actions, final_codes, final_ids.view(-1).mean()

    def process_exp_dataset_no_balance(self, inputs, action):

        input_code = torch.cat((inputs, action[:, :self.half_action_space]), dim=1)
        pred_code = self.recode(input_code).detach()

        return inputs, action, pred_code

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, code_size=2, base_net_small=False):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.hidden_size = hidden_size
        self.code_size = code_size
        self.base_net_small = base_net_small

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        if self.base_net_small:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs+self.code_size, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs+self.code_size, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        else:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs+self.code_size, hidden_size)), nn.ReLU(),
                init_(nn.Linear(hidden_size, hidden_size*2)), nn.ReLU(),
                init_(nn.Linear(hidden_size*2, hidden_size*2)), nn.Tanh(),
                init_(nn.Linear(hidden_size*2, hidden_size)), nn.Tanh())

            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs+self.code_size, hidden_size)), nn.ReLU(),
                init_(nn.Linear(hidden_size, hidden_size*2)), nn.ReLU(),
                init_(nn.Linear(hidden_size*2, hidden_size*2)), nn.Tanh(),
                init_(nn.Linear(hidden_size*2, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, random_seed, rnn_hxs, masks):
        x = inputs

        x = torch.cat((x, random_seed), dim=1)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
