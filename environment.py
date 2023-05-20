import numpy as np
import gym
import random
import math

from sklearn.preprocessing import KBinsDiscretizer
import time
from typing import Tuple
import tensorflow as tf
import datetime

class Environment(object):
    """docstring for Environment"""
    def __init__(self, algo=None, render_mode=None):
        super(Environment, self).__init__()
        self.algo = algo
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.num_of_action = self.env.action_space.n  # number of actions in the environment
        self.num_of_observation = self.env.observation_space.shape[0]  # number of observation in the environment
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state_bins = (3, 6, 6, 12)

    def state_space_to_discrete(self, position, velocity, angle, angular_velocity)->tuple[int,...]:
        """Convert continuous state space into a discrete state for Q-Learning"""
        upper_bound = [self.env.observation_space.high[0], 20.0, self.env.observation_space.high[2], math.radians(50)]
        lower_bound = [self.env.observation_space.low[0], -20.0, self.env.observation_space.high[2], -math.radians(50)]
        binDis = KBinsDiscretizer(n_bins=self.state_bins, encode="ordinal", strategy="uniform")
        binDis.fit([lower_bound, upper_bound])
        binT = binDis.transform([[position, velocity, angle, angular_velocity]])
        return tuple(map(int, binT[0]))

    def env_reset(self):
        observation,_ = self.env.reset()
        if self.algo == "Q-Learning":
            return self.state_space_to_discrete(*observation)
        return observation

    def env_step(self, action):
        next_obs, reward, terminated, done, info = self.env.step(action)
        if self.algo == "Q-Learning":
            next_obs = self.state_space_to_discrete(*next_obs)
        return next_obs, reward, terminated, done, info

    def env_close(self):
        self.env.close()


