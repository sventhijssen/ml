import random

import numpy as np

"""
The Q-Learning algorithm using the Boltzmann action selection mechanism.
"""
class Player:

    def __init__(self, actions=2):
        self.epsilon = 1
        self.visits = np.zeros(actions)
        self.q_table = np.zeros(actions)

    def update_q_table(self, action, reward):
        self.visits[action] += 1
        self.q_table[action] += 0.0001 * (reward - self.q_table[action])

    def get_q_table(self):
        return self.q_table

    def set_q_table(self, q_table):
        self.q_table = q_table

    def get_action(self, k):
        rnd = random.random()
        # if rnd < self.get_probability_action(0, k):
        #     return 0
        # return 1
        # if rnd > self.epsilon/(k+1):
        #     return np.argmax(self.q_table[:])
        # else:
        #     return random.choice([0, 1])
        c = self.cumsum()
        for i in range(len(c)):
            if c[i] >= rnd:
                return i
        raise Exception

    def cumsum(self):
        c = [self.get_probability_action(0,0)]
        for i in range(1,3):
            c.append(c[i-1] + self.get_probability_action(i,0))
        return c

    def get_probability_action(self,action,k):
        temp = 0.1
        return (np.exp(self.q_table[action]/temp))/(np.exp(self.q_table[0]/temp) + np.exp(self.q_table[1]/temp) + np.exp(self.q_table[2]/temp))
        # prob = (1-self.epsilon/(k+1))*0.5
        # if self.q_table[action] > self.q_table[self.other_action(action)]:
        #     prob += (self.epsilon/(k+1))
        # return prob
