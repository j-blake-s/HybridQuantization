import torch
import torch.nn as nn
import torch.nn.functional as F
import lava.lib.dl.slayer as slayer

from lava.lib.dl.slayer.block.cuba import Conv as SpkConv
from lava.lib.dl.slayer.block.cuba import Dense as SpkDense

class SpkDrop(torch.nn.Dropout3d):
  def forward(self, input):
    input_shape = input.shape
    return F.dropout3d(
      input.reshape((input_shape[0], -1, 1, 1, input_shape[-1])),
      self.p, self.training, self.inplace
    ).reshape(input_shape) * (1-self.p)

class SNN(torch.nn.Module):

  ### Init ###
  def __init__(self):
    super().__init__()

    params = {
      'threshold'     : 1,
      'current_decay' : 0.3,
      'voltage_decay' : 0.25,
      'tau_grad'      : 0.01,
      'requires_grad' : True,
    }

    self.conv = nn.Sequential(
      SpkConv(params, 2, 4, kernel_size=3, stride=2, padding=1),
      SpkDrop(0.05),
      SpkConv(params, 4, 8, kernel_size=3, stride=2, padding=1),
      SpkDrop(0.05),
      SpkConv(params, 8, 16, kernel_size=3, stride=2, padding=1),
      SpkDrop(0.05),
      SpkConv(params, 16, 32, kernel_size=3, stride=2, padding=1),
      SpkDrop(0.05),
      SpkConv(params, 32, 64, kernel_size=3, stride=2, padding=1),
      SpkDrop(0.05),
      SpkConv(params, 64, 128, kernel_size=3, stride=2, padding=1),
      SpkDrop(0.05),
    )

    self.dense = nn.Sequential(
      SpkDense(params, 2*2*128, 2056),
      SpkDrop(0.05),
      SpkDense(params, 2056, 128),
      SpkDrop(0.05),
      SpkDense(params, 128, 11)
    )


  ### Forward Pass ###
  def forward(self, x):

    B, C, H, W, T = x.shape
    x = self.conv(x)

    x = x.reshape(B,-1,T)

    x = self.dense(x)

    return x

  
def get_model(args):
  model = SNN()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  error = slayer.loss.SpikeRate(true_rate=0.5, false_rate=0.05, reduction='sum').to(args.device)
  classer = slayer.classifier.Rate.predict

  return model, optimizer, error, classer
