import os
import numpy as np
import torch

import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from configs.exp1_config import get_args as get_args_exp1
from configs.exp2_config import get_args as get_args_exp2
from configs.exp3_config import get_args as get_args_exp3

from libs.envs.env_exp1 import gameEnv
from libs.datasets.dataset_exp1 import Game_dataset
from libs.envs.env_exp2 import igEnv as igEnv_exp2
from libs.datasets.dataset_exp2 import ig_dataset as ig_dataset_exp2
from libs.envs.env_exp3 import igEnv as igEnv_exp3
from libs.datasets.dataset_exp3 import ig_dataset as ig_dataset_exp3

from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

import matplotlib.pyplot as plt

import pybullet as p

def replay_evaluate_2d(eval_envs, actor_critic, env_name, seed, num_processes, device, data_file):
    obs, _ = eval_envs.reset()

    eval_episode_rewards = []

    obs, random_seed = eval_envs.reset()
    obs, random_seed = obs.to(device), random_seed.to(device)

    eval_recurrent_hidden_states = torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    succ = False

    while True:
        traj = data_file.readline()[:-1]
        traj = traj.split(' ')
        if len(traj) != 14:
            break

        with torch.no_grad():
            action = [0.0, 0.0]
            action[0] = round(float(traj[10]), 1)
            action[1] = round(float(traj[11]), 1)
            action = torch.FloatTensor(action).to(device).view(1, 2)

            random_seed = actor_critic.evaluate_code(obs, action)

            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                random_seed,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        action[0][0] = round(float(traj[10]), 1)
        action[0][1] = round(float(traj[11]), 1)

        obs, succ, done, infos, random_seed = eval_envs.step(action)
        obs, random_seed = obs.to(device), random_seed.to(device)

        if done[0]:
            break

    return float(succ)


def replay_evaluate_ig(eval_envs, actor_critic, env_name, seed, num_processes, device, selected_actions, selected_states, selected_joints):
    eval_envs.start_eval()

    obs, random_seed = eval_envs.reset(provide=True, provide_states=selected_states, provide_joints=selected_joints)
    obs, random_seed = obs.to(device), random_seed.to(device)

    eval_recurrent_hidden_states = None
    eval_masks = None

    for step in range(max(600, len(selected_actions))):
        if step >= len(selected_actions):
            step = len(selected_actions) - 1

        with torch.no_grad():
            human_action = torch.FloatTensor(selected_actions[step][:7]).to(device).view(1, 7)
            human_action[:, :3] *= 1000.0

            random_seed = actor_critic.evaluate_code(obs, human_action)
            random_seed = torch.clip(random_seed, -1.0, 1.0)
            # print("pred_code:", random_seed)

            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                random_seed,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

            action[:, :7] = human_action

        obs, succ, done, infos, random_seed = eval_envs.step(action)
        obs, random_seed = obs.to(device), random_seed.to(device)

        if done[0]:
            eval_envs.stop_eval()
            if succ[0][0] == 1.0:
                return 1.0
            else:
                return 0.0

    eval_envs.stop_eval()
    return 0.0


def main(sysargv):
    if sysargv[2] == 'cogail_exp1_2dfq':
        args = get_args_exp1()
    elif sysargv[2] == 'cogail_exp2_handover':
        args = get_args_exp2()
    else:
        args = get_args_exp3()

    if args.env_name == 'cogail_exp1_2dfq':
        envs = gameEnv(args)
        dataset = 'dataset-continuous-info-act'
        dataset_id = [i for i in range(60, 100)] + [i for i in range(140, 150)] + [i for i in range(190, 200)]
    elif args.env_name == 'cogail_exp2_handover':
        envs = igEnv_exp2(args)
        dataset = 'dataset_handover_eval_final'
        dataset_id = [i for i in range(1, 14)] + [i for i in range(15, 68)]
    else:
        envs = igEnv_exp3(args)
        dataset = 'dataset_cabinet_replay_eval'
        dataset_id = [i for i in range(1, 61)]

    device = torch.device("cuda:0" if args.cuda else "cpu")

    folder = 'trained_models/{0}'.format(args.env_name)

    model_list = os.listdir(folder)
    seed_list = []
    for item in model_list:
        if len(item.split('_')) > 3 and int(item.split('_')[4]) not in seed_list:
            seed_list.append(int(item.split('_')[4]))

    for seed_select in seed_list:

        x = []
        y = []

        for model_id in model_list:
            if model_id[-3:] != '.pt' or len(model_id.split('_')) <= 3 or int(model_id.split('_')[4]) != seed_select:
                continue

            x.append(int(model_id.split('_')[6]))

            count = 0.0
            score = 0.0

            actor_critic, obs_rms = torch.load(os.path.join("{0}/{1}".format(folder, model_id)), map_location=device)

            for selected_id in dataset_id:

                if args.env_name == 'cogail_exp1_2dfq':
                    data_file = open('../dataset/{0}/{1}.txt'.format(dataset, selected_id), 'r')
                    succ = replay_evaluate_2d(envs, actor_critic, args.env_name, args.seed, args.num_processes, device, data_file)
                else:
                    selected_actions = np.load('../dataset/{0}/{1}_actions.npy'.format(dataset, selected_id))
                    selected_states = np.load('../dataset/{0}/{1}_states.npy'.format(dataset, selected_id))
                    selected_joints = np.load('../dataset/{0}/{1}_joints.npy'.format(dataset, selected_id))
                    succ = replay_evaluate_ig(envs, actor_critic, args.env_name, args.seed, args.num_processes, device, selected_actions, selected_states, selected_joints)

                score += succ
                count += 1.0
            y.append(score / count)

            print(model_id, score / count)

        new_x, new_y = zip(*sorted(zip(x, y)))

        plt.plot(new_x, new_y)
        plt.title(seed_select)
        plt.ylim(0, 1)
        plt.xlim(0, 1000)
        plt.savefig('{0}/results_seed{1}.png'.format(folder, seed_select))


if __name__ == "__main__":
    main(sys.argv)