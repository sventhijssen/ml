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

    def get_q_table(self):
        return self.q_table

    def get_action(self, k):
        rnd = random.random()
        if rnd > self.epsilon/(k+1):
            return np.argmax(self.q_table[:])
        else:
            return random.choice([0, 1])

    def get_probability_action(self,action,k):
        prob = (1-self.epsilon/(k+1))*0.5
        if self.q_table[action] > self.q_table[self.other_action(action)]:
            prob += (self.epsilon/(k+1))
        return prob

    def other_action(self,action):
        if action == 0:
            return 1
        else:
            return 0
