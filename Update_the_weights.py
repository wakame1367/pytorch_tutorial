import torch
import torch.nn as nn
import torch.optim as optim
from define_network import Net

if __name__ == '__main__':
    _input = torch.randn(1, 1, 32, 32)
    target = torch.randn((1, 10))
    criterion = nn.MSELoss()

    net = Net()
    lr = 0.01
    optimizer = optim.SGD(net.parameters(), lr=lr)

    optimizer.zero_grad()
    output = net(_input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
