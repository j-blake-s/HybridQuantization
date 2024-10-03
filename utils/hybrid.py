import torch
import torch.nn as nn
import torch.nn.functional as F
from lava.lib.dl.slayer.block.cuba import Conv as SpkConv
from .accumulator import AccumulateConv as Accumulator

# def cuba_pool(params, stride=2, delay_shift=True):
#   return slayer.block.cuba.Pool(
#     params, 2, stride=stride, delay=False, delay_shift=False
#   )


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
    )

  def forward(self, x):
    return self.net(x)





class ANN(nn.Module):
  def __init__(self, interval=8):
    super().__init__()

    self.net = nn.Sequential(

      Accumulator(interval=interval),
      
      nn.Conv2d( 8,  16, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      
      nn.Conv2d(16,  32, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      
      nn.Conv2d(32,  64, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      
      nn.Conv2d(64,  128, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),

      nn.Flatten(),
      
      nn.Linear(4*4*128, 512),
      nn.ReLU(),
      
      nn.Linear(512, 128),
      nn.ReLU(),
      
      nn.Linear(128, 11)
    )  


  def forward(self, x):
    if self.training:
      return F.softmax(self.net(x), dim=-1)
    else:
      return self.net(x)

class HNN(torch.nn.Module):

  def __init__(self, interval=8):
    super().__init__()

    self.snn = SNN()
    self.ann = ANN(interval=interval)

  def forward(self, x):
    if self.training:
      return self.ann(self.snn(x))
    else:
      x = self.snn(x)
      qx = torch.quantize_per_tensor(x, scale=1, zero_point=0, dtype=torch.quint8)
      return self.ann(qx)

def get_model(args):
  model = HNN()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  error = torch.nn.CrossEntropyLoss().to(args.device)
  classer = lambda x: torch.argmax(x,axis=-1)

  return model, optimizer, error, classer
