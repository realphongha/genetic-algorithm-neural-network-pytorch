import torch
import torch.nn as nn
import torch.nn.functional as F


class XorNN(nn.Module):
    # chromosome encodes a solution for the problem xor calculation
    def __init__(self, configs):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 8)
        self.fc3 = nn.Linear(8, 2)

    def init_weights(self, mean=0.0, std=1.0):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights with a normal distribution (mean=0, std=1)
                nn.init.normal_(m.weight, mean=0.0, std=1.0)
                # Initialize biases with zeros
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(self.fc3(x), dim=-1)
        return x


if __name__ == "__main__":
    model = XorNN()
    prob = model(torch.tensor([1, 0]).float())
    res = prob.argmax(0).item()
    print(prob, res)

