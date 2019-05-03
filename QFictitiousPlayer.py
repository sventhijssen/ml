import random

import numpy as np

"""
The combined Q-Learning and fictitious algorithm.
"""
class QFictitiousPlayer:

    def __init__(self, actions=2):
        self.epsilon = 1
        self.nr_stages = 0
        self.visits = np.zeros(shape=(actions,actions))
        self.q_table = np.zeros(shape=(actions,actions))
        self.probabilities = np.zeros(shape=(actions,actions))

    def update_q_table(self, action, opp_action, reward):
        self.nr_stages += 1
        self.visits[action,opp_action] += 1
        self.q_table[action,opp_action] += 0.001 * (reward - self.q_table[action,opp_action])
        self.probabilities = self.visits / self.nr_stages

    def get_q_table(self):
        return self.q_table

    def set_q_table(self, q_table):
        self.q_table = q_table

    def get_action(self, k):
        rnd = random.random()
        # if rnd < self.get_probability_action(0, k):
        #     return 0
        # return 1
        #if rnd > self.epsilon/(k+1):
            #print('MAX')
            #print(self.max_action())
        #    return self.max_action()
        #else:
            #print('RAND')
        #    a = random.choice([0, 1, 2])
            #print(a)
        #    return a
        c = self.cumsum()
        for i in range(len(c)):
            if c[i] >= rnd:
                return i
        return random.choice([0,1,2])
        #raise Exception

    def cumsum(self):
        c = [self.get_expected_value(0)]
        for i in range(0,2):
            c.append(c[i] + self.get_expected_value(i))
        return c

    def get_probability_action(self,action):
        temp = 0.1
        return (np.exp(self.q_table[action,0]/temp)+np.exp(self.q_table[action,1]/temp)+np.exp(self.q_table[action,2]/temp))/\
                            (np.exp(self.q_table[0,0]/temp)+np.exp(self.q_table[1,0]/temp) + np.exp(self.q_table[2,0]/temp)
                            +np.exp(self.q_table[0,1]/temp) + np.exp(self.q_table[1,1]/temp) + np.exp(self.q_table[2,1]/temp)
                            +np.exp(self.q_table[0,2]/temp) + np.exp(self.q_table[1,2]/temp) + np.exp(self.q_table[2,2]/temp))
        # prob = (1-self.epsilon/(k+1))*0.5
        # if self.q_table[action] > self.q_table[self.other_action(action)]:
        #     prob += (self.epsilon/(k+1))
        # return prob

    def max_action(self):
        mx = max(self.get_expected_value(0),self.get_expected_value(1),self.get_expected_value(2))
        for i in range(0,2):
            if(mx == self.get_expected_value(i)):
                return i
        return 0

    def get_expected_value(self, action):
        return ((self.q_table[action,0] * self.probabilities[action,0])
            + (self.q_table[action,1] * self.probabilities[action,1]) +
            (self.q_table[action,2] * self.probabilities[action,2]))

    def exp_prob(self,action):
        return (self.get_expected_value(action)/(self.get_expected_value(0)+self.get_expected_value(1)+self.get_expected_value(2)))