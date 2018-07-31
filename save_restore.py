"""
Script for testing pytorch save and restore functions
# https://morvanzhou.github.io/tutorials/machine-learning/torch/3-04-save-reload/
"""

import torch
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

    net = Net(n_input, n_hidden, n_output)

    # Two methods to save network

    # Method1: save both instructure of net and net parameters
    #torch.save(net, "save.pkl")
    # Method2: save net parameters only
    # Check for net.state_dict()
    #torch.save(net.state_dict(), "state_dict_save.pkl")

    # Two methods for restore the net
    # Method1
    reload_net = torch.load("./save.pkl")
    # Method2
    reload_weights = torch.load("./state_dict_save.pkl")
    new_net = Net(n_input, n_hidden, n_output)
    new_net.load_state_dict(reload_weights)
