import torch
import torch.nn as nn

# Source: https://towardsdatascience.com/a-simple-starter-guide-to-build-a-neural-network-3c2cf07b8d7c

input_size = 1  # The image size = 28 x 28 = 784
hidden_size = 3  # The number of nodes at the hidden layer
num_classes = 3  # The number of output classes. In this case, from 0 to 9
num_episodes = 20
learning_rate = 0.001  # The speed of convergence


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


net = Net(input_size, hidden_size, num_classes)

net.cuda()  # Run code on GPU

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


for episode in range(num_episodes):
    optimizer.zero_grad()  # Initialize the hidden weight to all zeros
    outputs = net(images)  # Forward pass: compute the output class given a image
    loss = criterion(outputs, labels)  # Compute the loss: difference between the output class and the pre-given label
    loss.backward()  # Backward pass: compute the weight
    optimizer.step()  # Optimizer: update the weights of hidden nodes