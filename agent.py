# include paths for modules to import
import sys
sys.path.append('config/')
sys.path.append('utils/')

# import necessary project modules/variables
from config import LEARNING_RATE, GAMMA, EPSILON_MAX, EPSILON_MIN, EPSILON_DECAY
from utils import exploration_rate, learning_rate_update
from q_learning import QLearning

# import libraries
import random
import math
import numpy as np
import datetime
import tensorflow as tf


# define agent Class

class Agent(object):
    """docstring for Agent"""
    def __init__(self, env):
        super(Agent, self).__init__()
        self.env = env
        self.num_states = env.num_of_observation  # number of states
        self.num_actions = env.num_of_action  # num of actions
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.state_bins = env.state_bins # only for Q-Learning

        self.learning_rate = LEARNING_RATE
        self.gamma = GAMMA  # discount factor
        self.epsilon_max = EPSILON_MAX
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.epsilon = self.epsilon_max  # initial epsilon

        # Q Learning algorithm
        self.q_learning = QLearning(self.state_bins, self.num_actions)

        # define writer
        self.logdir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = tf.summary.create_file_writer(logdir=self.logdir, name="q_learning")

    def exploration_exploitation(self, state):
        # get random epsilon value for exploration/explotation
        rand_epsilon = random.uniform(0, 1)

        if rand_epsilon <= self.epsilon:
            # explore the environment
            return self.action_space.sample()
        else:
            # exploit the environment. return action with max q value
            return self.q_learning.exploit(state)

    # Learn Q values
    def learn_environment(self, state, action, reward, next_state, episode):
        """
        Formula: Q(state, action) = old q value + learning rate *(current reward + discount factor* optimal q value from next state-old_q_value)
        """
        self.learning_rate = learning_rate_update(episode)
        self.q_learning.learn_environment(s=state,a=action,r=reward,n_s=next_state,lr=self.learning_rate,d_f=self.gamma)
        self.epsilon = exploration_rate(episode)

    def train(self):
        for episode in range(10):
            # reset environment
            observation = self.env.env_reset()

            # run upto this number of step in each episodes
            ep_reward = 0
            done = False

            while done == False:
                action = self.exploration_exploitation(observation)
                next_observation, reward, terminated, done, info = self.env.env_step(action)
                self.learn_environment(observation, action, reward, next_observation, episode)
                ep_reward += reward
                observation = next_observation

            with self.writer.as_default():
                tf.summary.scalar("reward", ep_reward, step=(episode + 1))
                self.writer.flush()

            # if (episode + 1) == 10000:
            #     self.env.render(mode="human")

        self.env.env_close()


