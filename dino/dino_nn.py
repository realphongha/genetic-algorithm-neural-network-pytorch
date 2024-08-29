import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoNN(nn.Module):
    # chromosome for dino game bot
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        # [dino_h, obs_y, obs_h, dist, speed]
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 3)

    def init_weights(self, a=-1.0, b=1.0):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights with a normal distribution (mean=0, std=1)
                nn.init.uniform_(m.weight, a, b)
                nn.init.uniform_(m.bias, a, b)
                # Initialize biases with zeros
                # nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

