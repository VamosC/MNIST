import torch
import torch.nn as nn
import numpy as np


class Convnet(nn.Module):

    def __init__(self):
        super().__init__()

        # convolution
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # fully connected
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 400)
        x = self.fc(x)
        return x


# unit test
if __name__ == "__main__":
    print(Convnet())
