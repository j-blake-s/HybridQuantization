import torch
import lava.lib.dl.slayer as slayer

def cuba_conv(params, in_, out_, kernel_size=3, stride=1, padding=0):
  return slayer.block.cuba.Conv(
    params, in_, out_, kernel_size=kernel_size, stride=stride, 
    padding=padding, weight_scale=2, weight_norm=True, delay=False, 
    delay_shift=False,
  )

def cuba_pool(params, num=2, stride=2, delay_shift=True):
  return slayer.block.cuba.Pool(
    params, num, stride=stride, delay=False, delay_shift=False
  )

def cuba_dense(params, in_, out_):
  return slayer.block.cuba.Dense(
    params, in_, out_, weight_scale=2, weight_norm=True, delay_shift=False,
  )

class DeepSNN(torch.nn.Module):

  ### Init ###
  def __init__(self, args):
    super(DeepSNN, self).__init__()

    cuba_params = {
      'threshold'     : 1.25,
      'current_decay' : 0.25,
      'voltage_decay' : 0.03,
      'tau_grad'      : 0.03,
      'requires_grad' : True,
      'dropout'       : slayer.neuron.Dropout(p=0.05),
    }

    self.conv = torch.nn.ModuleList([
      cuba_conv(cuba_params, 2,  4),
      cuba_conv(cuba_params, 4, 8),
      cuba_conv(cuba_params, 8, 16),
      cuba_conv(cuba_params, 16, 32),
      cuba_conv(cuba_params, 32, 64),

    ])

    self.pool = torch.nn.ModuleList([
      cuba_pool(cuba_params),
      cuba_pool(cuba_params),
      cuba_pool(cuba_params),
      cuba_pool(cuba_params),
      cuba_pool(cuba_params),
    ]) 

    self.flatten = slayer.block.cuba.Flatten()

    self.dense = torch.nn.ModuleList([
      cuba_dense(cuba_params, 64*3*3, 512),
      cuba_dense(cuba_params, 512, 128),
      cuba_dense(cuba_params, 128, args.classes),
    ])


  ### Forward Pass ###
  def forward(self, x):

    for c,p in zip(self.conv,self.pool):
      x = p(c(x))

    x = self.flatten(x)

    for d in self.dense:
      x = d(x)

    # print(x.shape)
    return x

  
def get_model(args):
  model = DeepSNN(args)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum').to(args.device)
  classer = slayer.classifier.Rate.predict

  return model, optimizer, error, classer
