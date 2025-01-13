import torch
import torch.nn as nn

class AccumulateConv(nn.Module):
  def __init__(self, interval):
    super().__init__()
    self.layer = nn.Conv3d(in_channels=1, out_channels=1, 
                         kernel_size=(1,1,interval), stride=(1,1,interval),
                         padding=0, bias=False)
    self.layer.weight = torch.nn.Parameter(torch.ones_like(self.layer.weight), requires_grad=False)
  
  def forward(self, x):
    B, C, H, W, T = x.shape # Input should have this shape
    x = x.reshape(B,1,C*H,W,T) # Collapse channels into height dimension

    x = self.layer(x) # B C*H W T//interval

    x = x.reshape(B, -1, H, W) # B C' H W  
    return x