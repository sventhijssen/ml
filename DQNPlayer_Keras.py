from collections import namedtuple
from random import random

from keras.models import Sequential
from keras.layers import Dense

from numpy import np
# fix random seed for reproducibility

##########
# README #
##########
# Based on:
# - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# - https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/
# - https://www.practicalai.io/teaching-a-neural-network-to-play-a-game-with-q-learning/


class Net:
    def __init__(self, input_size, hidden_size, num_classes):
        self.model = Sequential()
        self.model.add(Dense(input_size, input_dim=input_size, activation='relu'))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(num_classes, activation='sigmoid'))

# Source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
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
        self.model.add(Dense(nr_actions, input_dim=nr_actions, activation='relu'))
        self.model.add(Dense(nr_actions, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.epsilon = 0.1
        self.discount = 0.9
        self.replay_memory_size = 100
        self.replay_memory = ReplayMemory(self.replay_memory_size)
        self.batch_size = 20

    def optimize_model(self):
        pass

    def update_replay_memory(self, action, reward):
        # <s, a, s', r> is a tuple. We omit s and s' since these are always the initial state
        self.replay_memory.push((action, reward))

        if len(self.replay_memory) > self.replay_memory_size:
            batch = self.replay_memory.sample(self.batch_size)
            training_x_data = []
            training_y_data = []

            for b in batch:
                q_table_row = []
                for i in range(self.nr_actions):
                    input_state_action = None  # TODO: What is this?
                    input_state_action[0] = 1  # TODO: again what?

                    q_table_row[i] = self.net.forward(input_state_action)

                updated_q_value = reward + self.discount * max(q_table_row)

                training_x_data.append(None) #TODO
                training_y_data.append(None) #TODO

            # TODO: Train data somehow
            optimizer.zero_grad()  # Intialize the hidden weight to all zeros
            outputs = net(images)  # Forward pass: compute the output class given a image
            loss = criterion(training_x_data, training_y_data)  # Compute the loss: difference between the output class and the pre-given label
            loss.backward()  # Backward pass: compute the weight
            optimizer.step()

    def get_action(self):
        rnd = random()
        if rnd > self.epsilon:  # Select random action
            return random.randint(0, self.nr_actions)
        else:
            q_values = []
            for i in range(len(self.actions)):
                q_values[i] = self.model.predict(self.actions[i])
            return np.argmax(q_values)  # Return action with maximum reward outcome
