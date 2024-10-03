import os
import torch
import numpy as np
from utils.data import DvsGesture, temporal_jitter,  spatial_jitter
import lava.lib.dl.slayer as slayer
from torch.utils.data import DataLoader
from utils.argparser import args_parser
from utils.qtrain import train, qtest


### Load Args ###
args = args_parser()

# Choose model
from utils.hybrid import get_model as hnn
from utils.acnn import get_model as acnn
from utils.snn import get_model as snn
get_model, model_desc = {
  "hnn" : (hnn, "Hybrid Model"),
  "acnn": (acnn,"Accumulate-CNN Model"),
  "snn" : (snn, "SNN Model"),
}[args.model]

# Create Save Directory
args.save_path = os.path.join(args.save_folder, args.save_name)
args.model_path = os.path.join(args.save_path, f'{args.model}.pkl')
args.log_path = os.path.join(args.save_path, "log.txt")
args.acc_path = os.path.join(args.save_path, "acc.csv")
args.model_log_path = os.path.join(args.save_path, "model_log.txt")

# Define device
if args.gpu:
  args.device = "cuda:" + str(args.core)
  import cupy as lib
else:
  args.device = "cpu"
  import numpy as lib

# Print Info
print("="*40)
print(f'[Output] Saving to {args.save_path}')
print(f'[Model]\t {model_desc}')
print(f'[Device] {args.device}')

### Load Model ###
model, optimizer, error, classer = get_model(args)
model.to(args.device)

### Load Data ###
def augment(x):
  cx = lib.asarray(x)
  cx = temporal_jitter(cx, max_shift=4, lib=lib)
  cx = spatial_jitter(cx, max_shift=20, lib=lib)
  return np.asarray(cx)

train_path = os.path.join(args.data_path,"train.npz")
print(f'Loading samples...',end="\r")
# training = DvsGesture(train_path,transform=None)
training = DvsGesture(train_path,transform=augment)
train_loader = DataLoader(dataset=training, batch_size=args.batch_size, shuffle=True, drop_last=True)
print(f'Found {len(training):,} training samples...') 

test_path = os.path.join(args.data_path,"test.npz")
testing = DvsGesture(test_path)
test_loader = DataLoader(dataset=testing, batch_size=args.batch_size, shuffle=True, drop_last=True)
print(f'Found {len(testing):,} testing samples...') 


model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
torch.ao.quantization.prepare_qat(model, inplace=True)

### Train ###
for epoch in range(args.epochs):
  print(" "*50,end="\r")
  print(f'Epoch [{epoch+1}/{args.epochs}]')
  model.to(args.device)
  train_acc = train(model, train_loader, optimizer, error, classer, args)
  if epoch > 3: model.apply(torch.ao.quantization.disable_observer)
  model.to("cpu")
  qmodel = torch.ao.quantization.convert(model.eval(), inplace=False)
  qmodel.eval()
  test_acc = qtest(qmodel, test_loader, classer, args)
  print(f'\033[F\rEpoch [{epoch+1}/{args.epochs}] Training: {train_acc:.2%}\tValidation: {test_acc:.2%}              ')
