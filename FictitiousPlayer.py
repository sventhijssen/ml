import numpy as np


class FictitiousPlayer:

    def __init__(self):
        self.nr_stages = 1
        self.visits = np.array([1, 0, 0])
        self.probabilities = np.array([1, 0, 0])

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
