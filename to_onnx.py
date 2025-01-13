from onnx_models.acnn import AccCnn
import torch

model = AccCnn(11, timesteps=16, interval=2, quant=False)

inp = torch.randn(1, 16, 128, 128)
onnx = torch.onnx.export(
    model, 
    inp, 
    'acnn2_13.onnx',
    opset_version=13)
# onnx.save("acnn2_opset_20.onnx")

