import torch
import torch.nn as nn
import torch.nn.functional as F
from lava.lib.dl.slayer.block.cuba import Conv as SpkConv
from .accumulator import AccumulateConv as Accumulator
from .util import SpkDrop

class SNN(nn.Module):
  def __init__(self):
    super().__init__()

    params = {
      'threshold'     : 1,
      'current_decay' : 0.3,
      'voltage_decay' : 0.25,
      'tau_grad'      : 0.01,
      'requires_grad' : True
    }

    self.net = nn.Sequential(
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
    )

    self.output_channels = self.net[-1].synapse.weight.shape[0]

  def forward(self, x):
    return self.net(x)


class ANN(nn.Module):
  def __init__(self, in_channels, timesteps, interval):
    super().__init__()

    self.net = nn.Sequential(

      Accumulator(interval=interval),
      
      nn.Conv2d( in_channels*(timesteps//interval), 128, kernel_size=1, stride=1, padding=0),
      nn.ReLU(),
      
      nn.Flatten(),
      
      nn.Linear(2*2*128, 2056),
      nn.ReLU(),
      nn.Dropout(0.25),
      
      nn.Linear(2056, 128),
      nn.ReLU(),
      nn.Dropout(0.25),
      
      nn.Linear(128, 11)
    )  


  def forward(self, x):
    if self.training:
      return F.softmax(self.net(x), dim=-1)
    else:
      return self.net(x)

class HNN(torch.nn.Module):

  def __init__(self, timesteps, interval, quant=True):
    super().__init__()

    self.quant = quant
    self.snn = SNN()
    self.ann = ANN(self.snn.output_channels, timesteps, interval)

  def forward(self, x):
    x = self.snn(x)
    if self.training or not self.quant:
      return self.ann(x)
    else:
      qx = torch.quantize_per_tensor(x, scale=1, zero_point=0, dtype=torch.quint8)
      return self.ann(qx)

def get_model(args):
  model = HNN(timesteps=16, interval=args.interval, quant=(not args.no_quant))
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  error = torch.nn.CrossEntropyLoss().to(args.device)
  classer = lambda x: torch.argmax(x,axis=-1)

  return model, optimizer, error, classer