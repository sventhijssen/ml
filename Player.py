import random

import numpy as np


class Player:

    def __init__(self):
        self.epsilon = 1
        self.visits = np.zeros(2)
        self.q_table = np.zeros(2)

    def update_q_table(self, action, reward):
        self.visits[action] += 1
        self.q_table[action] += (1/(1+self.visits[action])) * (reward - self.q_table[action])

    def get_action(self, k):
        rnd = random.random()
        if rnd > self.epsilon/(k+1):
            return np.argmax(self.q_table[:])
        else:
            return random.choice([0, 1])

