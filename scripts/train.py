import os
import time
import random
import torch
import numpy as np
from collections import deque

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail

from configs.exp1_config import get_args as get_args_exp1
from configs.exp2_config import get_args as get_args_exp2
from configs.exp3_config import get_args as get_args_exp3

from libs.envs.env_exp1 import gameEnv
from libs.datasets.dataset_exp1 import Game_dataset
from libs.envs.env_exp2 import igEnv as igEnv_exp2
from libs.datasets.dataset_exp2 import ig_dataset as ig_dataset_exp2
from libs.envs.env_exp3 import igEnv as igEnv_exp3
from libs.datasets.dataset_exp3 import ig_dataset as ig_dataset_exp3

from scripts.evaluation import evaluate
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

def main(sysargv):
    if sysargv[2] == 'cogail_exp1_2dfq':
        args = get_args_exp1()
    elif sysargv[2] == 'cogail_exp2_handover':
        args = get_args_exp2()
    else:
        args = get_args_exp3()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.env_name == 'cogail_exp1_2dfq':
        envs = gameEnv(args)
    elif args.env_name == 'cogail_exp2_handover':
        envs = igEnv_exp2(args)
    else:
        envs = igEnv_exp3(args)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        args.recode_dim,
        base_kwargs={'recurrent': args.recurrent_policy,
                     'code_size': args.code_size,
                     'base_net_small': args.base_net_small})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 64,
            device, args.use_cross_entropy)

        if args.env_name == 'cogail_exp1_2dfq':
            expert_dataset = Game_dataset(args)
        elif args.env_name == 'cogail_exp2_handover':
            expert_dataset = ig_dataset_exp2(args)
        else:
            expert_dataset = ig_dataset_exp3(args)

        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.random_seed_space, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs, random_seed = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.random_seed[0].copy_(random_seed)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(args.bc_pretrain_steps):
        loss = agent.pretrain(gail_train_loader, device)
        print("Pretrain round {0}: loss {1}".format(j, loss))

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        if args.use_curriculum:
            envs.update_step_size(j, num_updates)

        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.random_seed[step],
                    rollouts.recurrent_hidden_states[step], rollouts.masks[step])

            obs, reward, done, infos, random_seed = envs.step(action)

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks, random_seed)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.random_seed[-1],
                rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()

        if args.gail:
            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 50  # Warm up

            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts, actor_critic)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], rollouts.random_seed[step], args.gamma,
                    rollouts.masks[step])

                episode_rewards.append(rollouts.rewards[step].item())
        else:
            for step in range(args.num_steps):
                episode_rewards.append(rollouts.rewards[step].item())

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, code_loss, inv_loss = agent.update(rollouts, gail_train_loader, device)

        rollouts.after_update()

        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.env_name)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.3f}/{:.3f}, min/max reward {:.3f}/{:.3f}, dist_ent {:.3f}, value loss {:.3f}, action loss {:.3f}, code loss {:.3f}, inv loss {:.3f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss, code_loss, inv_loss))

        if (args.eval_interval is not None and j % args.eval_interval == 0 and j != 0):
            eval_score = evaluate(envs, actor_critic, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device, 10)

            save_path = os.path.join(args.save_dir, args.env_name)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + '_seed_' + str(args.seed) + '_step_' + str(j) + "_score_" + str(eval_score) + ".pt"))


if __name__ == "__main__":
    main(sys.argv)

