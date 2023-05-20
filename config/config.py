# import necessary libraries
import os
from pathlib import Path


# set directories
ROOT_DIR = os.getcwd() # replace with the root directory of the project


# set the hyperparameters for the agent
LEARNING_RATE = 0.0001
GAMMA = 0.99 # discount factor
EPSILON_MAX = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.005
