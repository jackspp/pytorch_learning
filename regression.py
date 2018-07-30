"""
Reimplement from Mofan pytorch tutorial
# https://morvanzhou.github.io/tutorials/machine-learning/torch/3-01-regression/
"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# make some points x, y
x = torch.linspace(-1,1, 200)

# x shape should be (200,1)
x = torch.unsqueeze(x, dim=1)

# make some predicted y
y = x.pow(2) + 0.2 * torch.rand(x.size())

# X,Y should be variable
x,y = Variable(x), Variable(y)


# Visualize x,y via plt.scatter
#plt.scatter(x,y)
#plt.show()


# Define a NN
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        # Define layers of the NN
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))

        # Because the output has no specific range, no activation function is
        # used.
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
print net

# Optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

# Show info about training
plt.ion()
plt.show()

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.7, 0, 'Loss=%.4f' %(loss.data[0]), fontdict={"size":10})
        plt.pause(0.1)

plt.ioff()
plt.show()

