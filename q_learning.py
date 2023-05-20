import numpy as np
class QLearning:
    def __int__(self, state_bins, num_action=2):
        self.q_table = np.zeros(state_bins, (num_action,))

    def learn_environment(self, s, a, r, n_s, lr,d_f):
        """
        Learning the environment to balance the CartPole
        :param s: current state
        :param a: action
        :param r: reward
        :param n_s: next state
        :param lr: learning rate
        :param d_f: discount factor
        :return: q_table
        """
        self.q_table[s][a] = self.q_table[s][a]+lr*(r+d_f*np.max(self.q_table[n_s])-self.q_table[s][a])
        return self.q_table

    def exploit(self, state):
        """
        Return an optimal action based on current states
        :param state: current observation from the environment
        :return: current optimal action
        """
        return np.argmax(self.q_table[state])