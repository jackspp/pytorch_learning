"""
Optimizer test
# https://morvanzhou.github.io/tutorials/machine-learning/torch/3-06-optimizer/
"""

import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer


# make some points x, y
x = torch.linspace(-1,1, 1000)

# x shape should be (200,1)
x = torch.unsqueeze(x, dim=1)

# make some predicted y
y = x.pow(2) + 0.1 * torch.rand(x.size())

# X,Y should be variable
#x,y = Variable(x), Variable(y)

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

# Hyperparameters
LR = 0.01
MOMENTUM = 0.8
BATCH_SIZE = 32
EPOCH = 12


# Define loss function
loss_func = torch.nn.MSELoss()

# Define diffrent nets
sgd_net = Net(1, 10, 1)
momentum_net = Net(1, 10, 1)
adam_net = Net(1, 10, 1)
adagrad_net = Net(1, 10, 1)
net_list = [sgd_net, momentum_net, adam_net, adagrad_net]

# Define different optimizers
sgd_optim = torch.optim.SGD(sgd_net.parameters(),
                            lr=LR)
momentum_optim = torch.optim.SGD(momentum_net.parameters(),
                                 lr=LR,
                                 momentum=MOMENTUM)
adam_optim = torch.optim.Adam(adam_net.parameters(),
                             lr=LR,)
adagrad_optim = torch.optim.Adagrad(adagrad_net.parameters(),
                                    lr=LR)
optim_list = [sgd_optim, momentum_optim, adam_optim, adagrad_optim]

# Loss history
l_history =[[],[],[],[]]

# Define method to load data
tensor_data = Data.TensorDataset(x,y)
loader = Data.DataLoader(tensor_data,
                         batch_size=BATCH_SIZE,
                         shuffle=True)


# Training
for epoch in range(EPOCH):
    print "EPOCH: %d"%epoch
    for step, (batch_x, batch_y) in enumerate(loader):
        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y)
        print "step: %d"%step
        for net, optimizer, l_net_history in zip(net_list, optim_list, l_history):
            # Predcition
            prediction = net(batch_x)
            # loss
            batch_loss = loss_func(prediction, batch_y)
            # Optim
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # to numpy
            batch_loss = batch_loss.data.numpy()
            l_net_history.append(batch_loss)

# Plot
fig = plt.figure()
axis = fig.add_subplot(111)
axis.set_xlabel("Steps")
axis.set_ylabel("Loss")
axis.set_ylim(0, 0.2)
plt.plot(l_history[0], label="sgd")
plt.plot(l_history[1], label="momentum")
plt.plot(l_history[2], label="adam")
plt.plot(l_history[3], label="adagrad")
plt.legend(loc=0)
plt.show()












