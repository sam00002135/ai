import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #   origin
        #   1, 6
        #   6, 16
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=48, kernel_size=5)

        #   origin
        #   16*5*5  120
        #   120     84
        #   84      10
        self.fc1 = nn.Linear(in_features=48*5*5, out_features=240)
        self.fc2 = nn.Linear(in_features=240, out_features=96)
        self.fc3 = nn.Linear(in_features=96, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # x = x.view(-1, 16*5*5)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training) # new!
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training) # new!
        x = self.fc3(x)
        return x
