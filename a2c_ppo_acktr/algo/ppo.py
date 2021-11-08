import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.half_action_size = actor_critic.half_action_space

        self.MSEloss = nn.MSELoss()

    def pretrain(self, gail_train_loader, device):
        all_loss = []
        for expert_batch in gail_train_loader:
            expert_state, expert_action, _ = expert_batch
            expert_state = torch.FloatTensor(expert_state).to(device)
            expert_action = expert_action.to(device)

            bs = len(expert_state)
            expert_seed = torch.FloatTensor(np.array([[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(bs)])).view(bs, 2).to(device)

            pred_act = self.actor_critic.act(expert_state, expert_seed, None, None, deterministic=True)[1]
            loss = self.MSEloss(expert_action, pred_act)
            all_loss.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return np.mean(all_loss)

    def update(self, rollouts, expert_loader, device):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        code_loss_epoch = 0
        inv_loss_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for expert_batch, sample in zip(expert_loader, data_generator):
                obs_batch, random_seed_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, pred_code = self.actor_critic.evaluate_actions(
                    obs_batch, random_seed_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                code_loss = torch.norm(pred_code - random_seed_batch, dim=1).mean() * 0.1

                expert_state, expert_action, expert_id = expert_batch
                expert_state = torch.FloatTensor(expert_state).to(device)
                expert_action = expert_action.to(device)
                pred_codes = self.actor_critic.evaluate_code(expert_state, expert_action)
                _, pred_action, _, _ = self.actor_critic.act(expert_state, pred_codes, None, None, deterministic=True)
                inv_loss = torch.norm(expert_action[:, :self.half_action_size] - pred_action[:, :self.half_action_size], dim=1).mean() * 0.1

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss + code_loss + inv_loss -
                 dist_entropy * self.entropy_coef).backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                code_loss_epoch += code_loss.item()
                inv_loss_epoch += inv_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        code_loss_epoch /= num_updates
        inv_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, code_loss_epoch, inv_loss_epoch
