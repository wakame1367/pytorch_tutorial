"""
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
"""

import torch
import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input, out_channel, 3*3 conv
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 6 input, out_channel, 3 * 3
        self.conv2 = nn.Conv2d(6, 16, 3)
        # input, output
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        # input, num_classes
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # pooling window 2*2
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        # numpy.reshape
        # 16 * 6 * 6
        x = x.view(-1, self.num_flat_features(x))
        # activation function
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # size = shape
        # (1, 16, 6, 6) -> (16, 6, 6)
        size = x.size()[1:]
        num_features = 1
        # 16 * 6 * 6
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    net = Net()
    print(net)

    _input = torch.randn(1, 1, 32, 32)
    output = net(_input)
    print(output)

    net.zero_grad()
    print(output.backward(torch.randn(1, 10)))
