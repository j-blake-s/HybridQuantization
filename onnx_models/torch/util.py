import torch
import torch.nn.functional as F

class SpkDrop(torch.nn.Dropout3d):
  def forward(self, input):
    input_shape = input.shape
    return F.dropout3d(
      input.reshape((input_shape[0], -1, 1, 1, input_shape[-1])),
      self.p, self.training, self.inplace
    ).reshape(input_shape) * (1-self.p)