import os
import argparse
from collections import deque

import gym
import torch
import numpy as np

from src.sac.sac_model import SACQNet, SACActor
from src.sac.sac_policy import SACPolicy

def main(args):
    env = gym.make(args.env)
    obs = env.reset()
    env.render()
    action_space = env.action_space

    if torch.cuda.is_available() and args.gpu:  
        device = "cuda:0"
    else:
        if args.gpu:
            print("GPU flag set, but no GPU found! Using CPU.")
        device = "cpu"

    actor = SACActor(state_size=env.observation_space,
                      action_space=env.action_space,
                      fc=(256, 256),
                      conv_size=None).to(device)
    assert args.actor_weights, "Weights to load must be specified"
    actor.load(args.actor_weights)
    policy = SACPolicy(action_space=env.action_space,
                       model=actor,
                       device=device)

    reward = 0
    for i in range(100000000):
        _obs = torch.tensor(obs[None, :], dtype=torch.float32).to(device)
        action = policy.step(_obs)[0]

        # Take step on env with action
        new_obs, rewards, done, infos = env.step(action)
        reward += rewards
        env.render()
        obs = new_obs

        if done:
            print(f"Reward: {reward}")
            reward = 0
            input()
            obs = env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train PPO')

    parser.add_argument(
        "--actor_weights",
        type=str,
        default=None)
    parser.add_argument(
        "--env",
        type=str,
        default="BipedalWalker-v3")
    parser.add_argument(
        '--gpu',
        default=False,
        action='store_true')
    args = parser.parse_args()

    #logging.getLogger().setLevel(logging.INFO)
    main(args)

