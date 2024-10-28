import torch
import lava.lib.dl.slayer as slayer
import torch
import torch.nn as nn
import torch.nn.functional as F

from .accumulator import AccumulateConv


class AccCnn(torch.nn.Module):
  def __init__(self, num_classes, timesteps=16, interval=8):
    super(AccCnn, self).__init__()

    self.accumulator = AccumulateConv(interval)    

    # Conv Layers
    self.convs = torch.nn.ModuleList([
      nn.Conv2d(2*(timesteps//interval), 4, kernel_size=3, stride=2, padding=1),
      nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
      nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
      nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
      nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    ])

    # Dense Layers
    self.dense = torch.nn.ModuleList([
      nn.Linear(2*2*128, 2056),
      nn.Linear(2056, 128),
    ])

    # Output layer
    self.output = nn.Linear(self.dense[-1].out_features, num_classes)
    self.dropout = nn.Dropout(0.25)


  def forward(self, x):

    x = self.accumulator(x)

    for conv in self.convs[:]: 
      x = conv(x)
      x = F.relu(x)

    x = torch.flatten(x, 1)

    for d in self.dense:
      x = d(x)
      x = F.relu(x)
      x = self.dropout(x)

    x = self.output(x)
    x = F.relu(x)

    if self.training:
      x = F.softmax(x, dim=1)
    return x 



def get_model(args):
  model = AccCnn(args.classes, timesteps=16, interval=args.interval)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  error = torch.nn.CrossEntropyLoss().to(args.device)
  classer = lambda x: torch.argmax(x,axis=-1)
  return model, optimizer, error, classer
