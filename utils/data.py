import numpy as np
from torch.utils.data import Dataset


def h_flip(image, lib=np):
  if np.random.rand(1) < 0.5:
    return lib.flip(image, axis=-2)
  return image

def temporal_jitter(image, max_shift=3, lib=np):
  dt = np.random.randint(-max_shift, high=max_shift+1)
  temp = lib.zeros_like(image)
  if    dt < 0: temp[:,:,:,:dt] = image[:,:,:,-dt:]
  elif  dt > 0: temp[:,:,:,dt:] = image[:,:,:,:-dt]
  else: return image
  return temp

def spatial_jitter(image, max_shift=10, lib=np):
  dh = np.random.randint(-max_shift, high=max_shift+1)
  dw = np.random.randint(-max_shift, high=max_shift+1)
  
  _, H, W, _ = image.shape
  temp = lib.zeros_like(image)

  def idxs(shift, max_idx):
    if shift >= 1: return  (shift, max_idx), (0, max_idx-shift)
    elif shift==0: return (0,max_idx),  (0,max_idx)
    else: return      (0,max_idx+shift),(-shift, max_idx)

  (ihl, ihr), (thl, thr) = idxs(dh, H)
  (iwl, iwr), (twl, twr)  = idxs(dw, W)
  
  temp[:,thl:thr,twl:twr,:] = image[:,ihl:ihr,iwl:iwr,:]
  return temp

class Wrapper(Dataset):
  def __init__(self,x,y, transform=None):
    super(Wrapper, self).__init__()
    self.images = x
    self.labels = y
    self.augment = transform
  def __len__(self): return self.images.shape[0]
  def __getitem__(self,idx): 
    image = self.images[idx]
    label = self.labels[idx]
    if self.augment is not None:
      image = self.augment(image)
    return image.astype(np.float32), label

def DvsGesture(path, transform=None):

  # Load Dataset
  with np.load(path) as data:
    # images = np.expand_dims(data['x'],axis=1)
    images = data['x']
    labels = data['y'].astype(int)


  return Wrapper(images, labels, transform=transform)