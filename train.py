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
from models.s6a0 import get_model as s6a0
from models.s5a1 import get_model as s5a1
from models.s4a2 import get_model as s4a2
from models.s3a3 import get_model as s3a3
from models.s2a4 import get_model as s2a4
from models.s1a5 import get_model as s1a5
from models.snn import get_model as snn
from models.acnn import get_model as acnn

get_model, model_desc = {
  "acnn": (acnn,"Accumulate-CNN Model"),
  "s1a5" : (s1a5, "S1A5 Hybrid Model"),
  "s2a4" : (s2a4, "S1A5 Hybrid Model"),
  "s3a3" : (s3a3, "S1A5 Hybrid Model"),
  "s4a2" : (s4a2, "S1A5 Hybrid Model"),
  "s5a1" : (s5a1, "S1A5 Hybrid Model"),
  "s6a0" : (s6a0, "S1A5 Hybrid Model"),
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
# print("="*40)
# print(f'[Output] Saving to {args.save_path}')
# print(f'[Model]\t {model_desc}')
# print(f'[Device] {args.device}')

### Load Model ###
model, optimizer, error, classer = get_model(args)
model.to(args.device)


print(model)
quit()

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
