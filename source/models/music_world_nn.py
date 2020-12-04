
from torch import nn

class MusicWorldNN(nn.Module):

  def __init__(self):
    super().__init__()

    self.lin1 = nn.Linear(8, 64)
    self.act1 = nn.ReLU()

    self.lin2 = nn.Linear(64, 32)
    self.act2 = nn.ReLU()

    self.lin3 = nn.Linear(32, 8)

  def forward(self, x):
    x = self.lin1(x)
    x = self.act1(x)

    x = self.lin2(x)
    x = self.act2(x)

    x = self.lin3(x)

    return x