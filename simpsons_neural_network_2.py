import torch.nn as nn
import torch.nn.functional as F


# compared to SimpsonsNet1, we add another convolutional layer (conv4)
# we add batch normalization after each convolutional layer, this can stabilize training

class SimpsonsNet2(nn.Module):
    def __init__(self):
        super(SimpsonsNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)  # New layer

        self.bn1 = nn.BatchNorm2d(16)  # New layer
        self.bn2 = nn.BatchNorm2d(32)  # New layer
        self.bn3 = nn.BatchNorm2d(64)  # New layer
        self.bn4 = nn.BatchNorm2d(128)  # New layer

        self.fc1 = nn.Linear(128 * 14 * 14, 512)  # Changed dimensions
        self.fc2 = nn.Linear(512, 256)  # New layer
        self.fc3 = nn.Linear(256, 29)

        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # New layer

        x = x.view(-1, 128 * 14 * 14)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))  # New layer
        x = self.fc3(x)

        return x
