from collections import namedtuple
from random import random

import torch
import torch.nn as nn
from tensorflow import zeros

##########
# README #
##########
# Based on:
# - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# - https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/
# - https://www.practicalai.io/teaching-a-neural-network-to-play-a-game-with-q-learning/


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


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
            action = zeros(nr_actions)
            action[i] = 1
            self.actions.append(action)

        self.nr_actions = nr_actions
        self.net = Net(nr_actions, nr_actions, 1)
        self.epsilon = 0.1
        self.replay_memory_size = 100
        self.replay_memory = ReplayMemory(self.replay_memory_size)
        self.batch_size = 20

    def optimize_mode(self):
        pass

    def update_replay_memory(self, action, reward):
        # <s, a, s', r> is a tuple. We omit s and s' since these are always the initial state
        self.replay_memory.push((action, reward))

        self.optimize_model()

    def get_action(self):
        rnd = random()
        if rnd > self.epsilon:  # Select random action
            return random.randint(0, self.nr_actions)
        else:
            q_values = []
            for i in range(len(self.actions)):
                q_values[i] = None  # TODO: run NN for given action
            return torch.argmax(q_values)  # Return action with maximum reward outcome



# class QLearningPlayer
#   attr_accessor :x, :y, :game
#
#   def initialize
#     @x = 0
#     @y = 0
#     @actions = [:left, :right, :up, :down]
#     @first_run = true
#
#     @discount = 0.9
#     @epsilon = 0.1
#     @max_epsilon = 0.9
#     @epsilon_increase_factor = 800.0
#
#     @replay_memory_size = 500
#     @replay_memory_pointer = 0
#     @replay_memory = []
#     @replay_batch_size = 400
#
#     @runs = 0
#
#     @r = Random.new
#   end
#
#   def initialize_q_neural_network
#     # Setup model
#     # Input is the size of the map + number of actions
#     # Output size is one
#     @q_nn_model = RubyFann::Standard.new(
#                   num_inputs: @game.map_size_x*@game.map_size_y + @actions.length,
#                   hidden_neurons: [ (@game.map_size_x*@game.map_size_y+@actions.length) ],
#                   num_outputs: 1 )
#
#     @q_nn_model.set_learning_rate(0.2)
#
#     @q_nn_model.set_activation_function_hidden(:sigmoid_symmetric)
#     @q_nn_model.set_activation_function_output(:sigmoid_symmetric)
#
#   end
#
#   def get_input
#     # Pause to make sure humans can follow along
#     # Increase pause with the number of runs
#     sleep 0.05 + 0.01*(@runs/400.0)
#     @runs += 1
#
#     if @first_run
#       # If this is first run initialize the Q-neural network
#       initialize_q_neural_network
#       @first_run = false
#     else
#       # If this is not the first
#       # Evaluate what happened on last action and calculate reward
#       r = 0 # default is 0
#       if !@game.new_game and @old_score < @game.score
#         r = 1 # reward is 1 if our score increased
#       elsif !@game.new_game and @old_score > @game.score
#         r = -1 # reward is -1 if our score decreased
#       elsif !@game.new_game
#         r = -0.1
#       end
#
#       # Capture current state
#       # Set input to network map_size_x * map_size_y + actions length vector with a 1 on the player position
#       input_state = Array.new(@game.map_size_x*@game.map_size_y + @actions.length, 0)
#       input_state[@x + (@game.map_size_x*@y)] = 1
#
#       # Add reward, old_state and input state to memory
#       @replay_memory[@replay_memory_pointer] = {reward: r, old_input_state: @old_input_state, input_state: input_state}
#       # Increment memory pointer
#       @replay_memory_pointer = (@replay_memory_pointer<@replay_memory_size) ? @replay_memory_pointer+1 : 0
#
#       # If replay memory is full train network on a batch of states from the memory
#       if @replay_memory.length > @replay_memory_size
#         # Randomly samply a batch of actions from the memory and train network with these actions
#         @batch = @replay_memory.sample(@replay_batch_size)
#         training_x_data = []
#         training_y_data = []
#
#         # For each batch calculate new q_value based on current network and reward
#         @batch.each do |m|
#           # To get entire q table row of the current state run the network once for every posible action
#           q_table_row = []
#           @actions.length.times do |a|
#             # Create neural network input vector for this action
#             input_state_action = m[:input_state].clone
#             # Set a 1 in the action location of the input vector
#             input_state_action[(@game.map_size_x*@game.map_size_y) + a] = 1
#             # Run the network for this action and get q table row entry
#             q_table_row[a] = @q_nn_model.run(input_state_action).first
#           end
#
#           # Update the q value
#           updated_q_value = m[:reward] + @discount * q_table_row.max
#
#           # Add to training set
#           training_x_data.push(m[:old_input_state])
#           training_y_data.push([updated_q_value])
#         end
#
#         # Train network with batch
#         train = RubyFann::TrainData.new( :inputs=> training_x_data, :desired_outputs=>training_y_data );
#         @q_nn_model.train_on_data(train, 1, 1, 0.01)
#       end
#     end
#
#     # Capture current state and score
#     # Set input to network map_size_x * map_size_y vector with a 1 on the player position
#     input_state = Array.new(@game.map_size_x*@game.map_size_y + @actions.length, 0)
#     input_state[@x + (@game.map_size_x*@y)] = 1
#     # Chose action based on Q value estimates for state
#     # If a random number is higher than epsilon we take a random action
#     # We will slowly increase @epsilon based on runs to a maximum of @max_epsilon - this encourages early exploration
#     epsilon_run_factor = (@runs/@epsilon_increase_factor) > (@max_epsilon-@epsilon) ? (@max_epsilon-@epsilon) : (@runs/@epsilon_increase_factor)
#     if @r.rand > (@epsilon + epsilon_run_factor)
#       # Select random action
#       @action_taken_index = @r.rand(@actions.length)
#     else
#       # To get the entire q table row of the current state run the network once for every posible action
#       q_table_row = []
#       @actions.length.times do |a|
#         # Create neural network input vector for this action
#         input_state_action = input_state.clone
#         # Set a 1 in the action location of the input vector
#         input_state_action[(@game.map_size_x*@game.map_size_y) + a] = 1
#         # Run the network for this action and get q table row entry
#         q_table_row[a] = @q_nn_model.run(input_state_action).first
#       end
#       # Select action with highest posible reward
#       @action_taken_index = q_table_row.each_with_index.max[1]
#     end
#
#     # Save score and current state
#     @old_score = @game.score
#
#     # Set action taken in input state before storing it
#     input_state[(@game.map_size_x*@game.map_size_y) + @action_taken_index] = 1
#     @old_input_state = input_state
#
#     # Take action
#     return @actions[@action_taken_index]
#   end
#
#
#   def print_table
#     @q_table.length.times do |i|
#       puts @q_table[i].to_s
#     end
#   end
#
# end