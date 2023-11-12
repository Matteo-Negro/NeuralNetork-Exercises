class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.max_pool0 = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 8, 5)
        self.max_pool1 = nn.MaxPool2d(5, 5)

        self.conv2 = nn.Conv2d(8, 16, 5)
        self.max_pool2 = nn.MaxPool2d(7, 7)

        self.lin1 = nn.Linear(64, 128)
        self.lin2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.max_pool0(x)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)

        x = torch.flatten(x, 1)

        x = self.lin1(x)
        x = F.relu(x)

        x = self.lin2(x)

        return x