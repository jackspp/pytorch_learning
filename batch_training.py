"""
Script for testing batch training of pytorch
# https://morvanzhou.github.io/tutorials/machine-learning/torch/3-05-train-on-batch/
"""

import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np
from IPython.core.debugger import Tracer

class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.input_layer = torch.nn.Linear(n_input, n_hidden)
        self.hidden_layer = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden_layer(x)
        return x

if __name__ == "__main__":
    # initialize a network
    # Parameters
    n_input = 2
    n_hidden = 10
    n_output = 2
    batch_size = 5

    net = Net(n_input, n_hidden, n_output)

    # Batch data
    x = torch.linspace(-1, 1, 100)
    y = torch.linspace(1, -1, 100)

    torch_dataset = Data.TensorDataset(x,y)


    loader = Data.DataLoader(torch_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=3)

    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            print step,"|","batch_x:", batch_x, "\n", "|", "batch_y:",batch_y, "\n"
