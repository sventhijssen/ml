from collections import namedtuple
from random import randint, random

from keras.models import Sequential
from keras.layers import Dense

import numpy as np

##########
# README #
##########
# Based on:
# - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# - https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/
# - https://www.practicalai.io/teaching-a-neural-network-to-play-a-game-with-q-learning/


Transition = namedtuple('Transition', ('action', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNPlayer:
    def __init__(self, nr_actions=3):
        self.actions = []  # Create vectors for the actions
        for i in range(nr_actions):
            action = np.zeros(nr_actions)
            action[i] = 1
            self.actions.append(action)

        self.nr_actions = nr_actions

        self.model = Sequential()
        self.model.add(Dense(nr_actions, input_shape=(nr_actions, ), activation='relu'))
        self.model.add(Dense(nr_actions, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.epsilon = 0.1
        self.discount = 0.9
        self.replay_memory_size = 100
        self.replay_memory = ReplayMemory(self.replay_memory_size)
        self.batch_size = 20

    def update_replay_memory(self, action, reward):
        # <s, a, s', r> is a tuple. We omit s and s' since these are always the initial state
        self.replay_memory.push(action, reward)

        if len(self.replay_memory) > self.replay_memory_size:
            batch = self.replay_memory.sample(self.batch_size)
            training_x_data = []
            training_y_data = []

            for m in batch:
                q_table_row = []
                for i in range(self.nr_actions):
                    q_table_row[i] = self.model.predict(self.actions[i])

                updated_q_value = m.reward + self.discount * max(q_table_row)

                training_x_data.append(m.action)
                training_y_data.append(updated_q_value)

            self.model.fit(training_x_data, training_y_data)

    def get_action(self):
        rnd = random()
        if rnd > self.epsilon:  # Select random action
            print("rnd")
            return randint(0, self.nr_actions-1)
        else:
            print("calc")
            q_table_row = []
            for i in range(len(self.actions)):
                q_table_row[i] = self.model.predict(np.transpose(self.actions[i]))
            return np.argmax(q_table_row)  # Return action with maximum reward outcome
