import torch.quantization
import torch


def dequant(qx):
  x = torch.zeros(size=qx.shape)

  if len(qx.shape) == 2:
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        x[i,j] = qx[i,j].item()

  elif len(qx.shape) == 3:
    for i in range(x.shape[0]):
      for j in range(x.shape[1]):
        for z in range(x.shape[2]):
          x[i,j] = qx[i,j,z].item()
  return x

def quantize_model(model, sample_data, backend='x86'):

  # Set qconfig
  model.qconfig = torch.quantization.get_default_qconfig(backend)

  # Place observers throughout model
  torch.quantization.prepare(model, inplace=True)

  # Collect value distribution from sample data
  for _, (x,_) in enumerate(sample_data):
    model.eval()
    out = model(x)
    break

  # Quantize model
  torch.quantization.convert(model, inplace=True)

  return model


def quantized_inference(qmodel, data, scale=1.0, zero_point=0):

  qmodel.eval()

  total_samples = 0
  total_correct_samples = 0
  with torch.no_grad():
    for i, (x,y) in enumerate(data):
      
      # Quantize input tensor
      qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=torch.quint8)

      # Run Model      
      qout = qmodel(qx)
      outputs = dequant(qout)

      # Get Accuracy
      total_samples += x.shape[0]
      correct_samples = torch.sum(torch.argmax(outputs,dim=1)==y).cpu().data.item()
      total_correct_samples += correct_samples

      acc = total_correct_samples / total_samples

      print(f'\rBatch [{i+1}/{len(data)}] Validation: {acc:.2%}',end="")
  print(f'\rValidation: {acc:.2%}                                   ')
  return acc


