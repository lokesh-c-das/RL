import math
import random
import os
from pathlib import Path
from collections import deque, namedtuple


# method to update exploration rate
def exploration_rate(epoch, min_rate=0.1):
    return max(min_rate, min(1, 1.0 - math.log10((epoch + 1) / 25)))


# method to update learning rate
def learning_rate_update(epoch, min_rate=0.01):
    return max(min_rate, min(1.0, 1.0 - math.log10((epoch + 1) / 25)))


# define a class for replay memory
class ReplayMemory(object):
    def __init__(self, memory_size):
        super(ReplayMemory, self).__init__()
        self.memory = deque([], maxlen=memory_size)

    # method to add samples in the replay memory
    def add_experience(self, *args):
        # create a new tuple Transition = ('state', 'action', 'next_state', 'reward')
        # that will be constituted from the return value of step function of the env
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.memory.append(Transition(*args))

    # sample from the memory space based on batch size
    def sample_experience(self, batch_size):
        return random.sample(self.memory, batch_size)

    # get the size of the memory
    def __len__(self):
        return len(self.memory)