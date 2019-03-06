import random

import numpy as np

# QTable
#   | H | T
# S |   |
class QLearning:

    def __init__(self, environment):
        self.alpha = 0.8
        self.gamma = 0.9
        self.epsilon = 1
        self.environment = environment
        self.action_size = environment.get_number_of_actions()
        self.state_size = environment.get_number_of_states()
        self.q_table = np.zeros(2)

    def reward(self, action, reward):
        action = int(action)
        self.q_table[action] = self.q_table[action] + self.alpha * (reward + self.gamma * np.max(self.q_table[:] - self.q_table[action]))

    def action(self):
        rnd = random.random()
        if rnd > self.epsilon:
            self.epsilon -= 0.1
            print('max'+ str(np.argmax(self.q_table[:])))
            return np.argmax(self.q_table[:])
        else:
            self.epsilon -= 0.1
            return self.q_table[random.choice([0, 1])]

