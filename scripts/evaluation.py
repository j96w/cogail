import numpy as np
import torch

def evaluate(eval_envs, actor_critic, env_name, seed, num_processes, eval_log_dir,
             device, total_round=10):

    eval_episode_rewards = []

    eval_envs.start_eval()

    obs, random_seed = eval_envs.reset()
    obs, random_seed = obs.to(device), random_seed.to(device)

    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < total_round:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                random_seed,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        obs, succ, done, infos, random_seed = eval_envs.step(action)
        obs, random_seed = obs.to(device), random_seed.to(device)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        if done[0]:
            if succ[0][0] == 1.0:
                eval_episode_rewards.append(1.0)
            else:
                eval_episode_rewards.append(0.0)

    eval_envs.stop_eval()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    return np.mean(eval_episode_rewards)
