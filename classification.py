"""
Reimplementation of easy classification task from Mofan video
# https://morvanzhou.github.io/tutorials/machine-learning/torch/3-02-classification/
"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer



# Define data
# x0
m_data = torch.ones(100 ,2)
x0 = torch.normal(2*m_data, 1)
y0 = torch.zeros(100)

# x1
x1 = torch.normal(-2*m_data, 1)
y1 = torch.ones(100)

# concatenent all the dataset
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1)).type(torch.LongTensor)
#y = torch.unsqueeze(torch.cat((y0, y1)).type(torch.FloatTensor), dim=-1)


x, y = Variable(x), Variable(y)

# Visualize dataset
#plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=y.data.numpy(), s=100, lw=0, )
#plt.show()

# Define Net
class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_input, n_hidden)
        self.output_layer = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

# Initialize network
net = Net(2, 10, 2)
print(net)

# Define optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.002)
loss_func = torch.nn.CrossEntropyLoss()

# initialize plot
plt.ion()
plt.show()

for t in range(500):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(prediction), 1)[1]
        plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=prediction.numpy())

        # Calculate accuracy
        accuracy = np.sum(prediction.data.numpy() == y.data.numpy()) / 200.0
        plt.text(3, -3, "Accuracy:%.4f" %(accuracy), fontdict={"size":10})
        plt.pause(0.1)

plt.ioff()
plt.show()








