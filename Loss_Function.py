import torch
import torch.nn as nn

from define_network import Net

if __name__ == '__main__':
    _input = torch.randn(1, 1, 32, 32)
    net = Net()
    output = net(_input)
    target = torch.randn((1, 10))
    # target = target.view(1, -1)

    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(loss)

    net.zero_grad()

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)
