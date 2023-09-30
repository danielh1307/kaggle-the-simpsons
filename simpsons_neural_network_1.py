import torch.nn as nn
import torch.nn.functional as F


class SimpsonsNet1(nn.Module):
    def __init__(self):
        super(SimpsonsNet1, self).__init__()
        # convolutional layer
        # Input: [3, 224, 224] (original image)
        # Output: [16, 224, 224]
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)

        # since we apply max pooling afterward with a 2x2 window, we'll get [16, 112, 112],
        # which is the input for the next convolutional layer
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # Output: [32, 112, 112]
        # after pooling with 2x2: [32, 56, 56], which is the input for the next convolutional layer

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # output: [64, 56, 56]
        # after pooling with 2x2: [64, 28, 28], which is the input for the next layer
        # flatten: 64 * 28 * 28 = 50176

        # Input: 64*28*28 (see above)
        # Output: 256
        self.fc1 = nn.Linear(64 * 28 * 28, 256)

        # Our last fully connected layer has 29 output features since we have 29 different labels.
        # Input: 256 (from previous layer)
        # Output: 29 for the 29 different labels
        self.fc2 = nn.Linear(256, 29)

        self.dropout = nn.Dropout(0.25)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    # The forward method defines the forward pass of the neural network, specifying how the input x is transformed into the output.
    def forward(self, x):
        # The First convolutional layer is applied.
        # ReLU (Rectified Linear Unit) activation function is applied.
        # Max pooling is applied.
        x = self.pool(F.relu(self.conv1(x)))

        # this is repeated for conv2 and conv3
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # The output tensor is reshaped (flattened) to feed into the fully connected layers. The -1 is a placeholder
        # for the batch size, and 64*4*4 is the number of features coming from the last convolutional layer.
        x = x.view(-1, 64 * 28 * 28)

        # Dropout is applied for regularization.
        x = self.dropout(x)

        # The First fully connected layer is applied.
        # ReLU's activation is applied.
        x = F.relu(self.fc1(x))

        # Dropout is applied again for regularization.
        x = self.dropout(x)

        # The Second fully connected layer is applied to produce the final output.
        x = self.fc2(x)

        # The final output is returned, which can be used for classification.
        return x
