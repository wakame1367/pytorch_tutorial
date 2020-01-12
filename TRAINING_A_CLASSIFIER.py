from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset_path = Path("resources")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input, out_channel, 3*3 conv
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 6 input, out_channel, 3 * 3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # input, output
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # input, num_classes
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # pooling window 2*2
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        # numpy.reshape
        # 16 * 6 * 6
        x = x.view(-1, 16 * 5 * 5)
        # activation function
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(epochs=10):
    batch_size = 4
    train_set = torchvision.datasets.CIFAR10(root=str(dataset_path),
                                             train=True,
                                             download=True,
                                             transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root=str(dataset_path),
                                            train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for idx, data in enumerate(train_loader, 1):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if idx % 2000 == 1999:
                print("[%d, %5d] loss: %.3f" %
                      (epoch, idx, total_loss / 2000))
                total_loss = 0.0
    print("train Finished")
    torch.save(net.state_dict(), dataset_path / "cifar_net.pth")


if __name__ == '__main__':
    train(epochs=2)
