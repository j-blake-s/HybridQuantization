import torch
import lava.lib.dl.slayer as slayer
from lava.lib.dl.slayer.block.cuba import Flatten
from lava.lib.dl.slayer.block.cuba import Dense
import torch
import torch.nn as nn
import torch.nn.functional as F

from .accumulator import AccumulateConv
from math import floor


class AccCnn(torch.nn.Module):
  def __init__(self, num_classes, accumulate_interval=8):
    super(AccCnn, self).__init__()

    self.accumulator = AccumulateConv(accumulate_interval)    

    # Conv Layers
    self.convs = torch.nn.ModuleList([
      nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    ])

    # Dense Layers
    self.dense = torch.nn.ModuleList([
      nn.Linear(2*2*256, 512),
      nn.Linear(512, 128),
    ])

    # Output layer
    self.output = nn.Linear(self.dense[-1].out_features, num_classes)
    self.dropout = nn.Dropout(0.25)


  def forward(self, x):

    x = self.accumulator(x)

    for conv in self.convs[:]: 
      x = conv(x)
      x = F.relu(x)
      x = F.max_pool2d(x, 2)

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
  model = AccCnn(args.classes)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  error = torch.nn.CrossEntropyLoss().to(args.device)
  classer = lambda x: torch.argmax(x,axis=-1)
  return model, optimizer, error, classer
