import gym
import torch
from collections import deque
import numpy as np

from src.gym_wrappers import *


def _format_observation(obs):
    obs = torch.squeeze(torch.as_tensor(obs))
    return obs.view((1, 1) + obs.shape)


def make_environment(flags, embedding_model, actor_id=1):
    seed = (flags.run_id + 1) * (actor_id + 1)
    gym_env = make_gym_env(flags, embedding_model, seed)
    return Environment(gym_env)


class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_success = None
        self.episode_step = None

    def render(self):
        self.gym_env.render()

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        self.episode_success = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.tensor(False, dtype=torch.bool).view(1, 1)
        self.gym_env.randomize()
        initial_obs = _format_observation(self.gym_env.reset())

        return dict(
            obs=initial_obs,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_success=self.episode_success,
            episode_step=self.episode_step,
            )

    def step(self, action):
        obs, reward, done, info = self.gym_env.step(action.item())
        success = info['success']

        self.episode_step += 1
        episode_step = self.episode_step

        self.episode_return += reward
        self.episode_success += success
        episode_return = self.episode_return
        episode_success = self.episode_success

        if done:
            self.gym_env.randomize()
            obs = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_success = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        obs = _format_observation(obs)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done, dtype=torch.bool).view(1, 1)

        return dict(
            obs=obs,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_success=episode_success,
            episode_step=episode_step,
            )

    def close(self):
        self.gym_env.close()
