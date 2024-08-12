import torch
import torch.nn as nn
import torch.nn.functional as F


class XorNN(nn.Module):
    # chromosome encodes a solution for the problem xor calculation
    def __init__(self, configs):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def init_weights(self, a=-1.0, b=1.0):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights with a normal distribution (mean=0, std=1)
                nn.init.uniform_(m.weight, a=a, b=b)
                nn.init.uniform_(m.bias, a=a, b=b)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

