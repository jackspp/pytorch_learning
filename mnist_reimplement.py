"""
Reimplementation of mnist classification using CNN
# https://morvanzhou.github.io/tutorials/machine-learning/torch/4-01-CNN/
"""

import numpy as np
import torch.utils.data as Data
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from IPython.core.debugger import Tracer

# Reproductable
torch.manual_seed(1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1,
                                                         out_channels=16,
                                                         kernel_size=5,
                                                         stride=1,
                                                         padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2, stride=2,))

        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=16,
                                                         out_channels=32,
                                                         kernel_size=5,
                                                         stride=1,
                                                         padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2, stride=2))

        self.linear_layer = torch.nn.Linear(7*7*32, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        middle_output =  x.view(x.size()[0], -1)
        x = self.linear_layer(middle_output)
        return x, middle_output


def load_mnist(data_path,
               train=True):

    DOWNLOAD = False
    if not os.path.exists(data_path):
        DOWNLOAD = True
    mnist_data = torchvision.datasets.MNIST(data_path,
                               train=train,
                               transform=torchvision.transforms.ToTensor(),
                               target_transform=None,
                               download=DOWNLOAD)

    return mnist_data



if __name__ == "__main__":

    # Paramters
    BATCHSIZE = 64
    EPOCH = 10
    LR = 0.05
    MOMENTUM = 0.8
    # Test or not
    # NOT USED!
    TEST = False

    # load mnist train dataset
    mnist_path = "./mnist"
    train_mnist = load_mnist(mnist_path,
                             train=True)
    test_mnist = load_mnist(mnist_path,
                            train=False)

    # Visualize one train img
    #data_index  = 2
    #plt.imshow(train_mnist.train_data[data_index].numpy(), cmap="gray")
    #print("Image label is: %d" %train_mnist.train_labels[data_index])
    #plt.show()


    # Make loader
    # Normalize trainning image is of great importance
    tensor_dataset = Data.TensorDataset(torch.unsqueeze(train_mnist.train_data, 1).type(torch.FloatTensor) / 255.0,
                                        train_mnist.train_labels)

    loader = Data.DataLoader(tensor_dataset,
                             batch_size = BATCHSIZE,
                             shuffle=True,
                             num_workers=3)


    # Initial Net
    conv_net = Net()

    # Loss func
    loss_func = torch.nn.CrossEntropyLoss()

    # Optimizer
    optim = torch.optim.SGD(conv_net.parameters(),
                            lr=LR,
                            momentum=MOMENTUM)


    # Training
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            #print "Epoch: %d | Step: %d" %(epoch, step)
            # Prediction
            prediction, middle_output =  conv_net(batch_x)
            # Loss
            batch_loss = loss_func(prediction, batch_y)
            # Optim
            optim.zero_grad()
            batch_loss.backward()
            optim.step()

            if step % 5 == 0:
                # Now use only 200 test dataset
                test_input = torch.unsqueeze(test_mnist.test_data[:200],
                                             1).type(torch.FloatTensor) / 255.0
                test_target = test_mnist.test_labels[:200]
                t_pred, t_middle_output = conv_net(test_input)
                t_accuracy = torch.sum(torch.max(t_pred, 1)[1] == test_target).numpy() / 200.0
                print("Accuracy: %f"%t_accuracy)

    # Training done
    print("Finished training!")












