import numpy as np
from random import randint

class FictitiousPlayer:

    def __init__(self):
        self.nr_stages = 0
        self.visits = np.array([0, 0, 0])
        self.probabilities = np.array([0, 0, 0])

    def set_action(self, action):
        self.nr_stages += 1
        self.visits[action] += 1
        self.probabilities = self.visits/self.nr_stages

    def get_action(self, payoff_matrix):
        utility = np.dot(payoff_matrix, np.transpose(self.probabilities))
        action = np.argmax(utility)
        return action

    def get_probabilities(self):
        return self.probabilities

    def set_probabilities(self,prob):
        self.probabilities = prob

    def set_visits(self,vis):
        self.visits = vis

    def set_stages(self,stage):
        self.nr_stages = stage
