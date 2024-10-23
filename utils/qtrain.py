

import torch
def train(model, data, opt, loss_fn, classifier, args):
  model.train()
  total_samples = 0
  loss_sum = 0
  total_correct_samples = 0
  for i, (images, labels) in enumerate(data):
    
    # Run Model #
    images = images.to(args.device)
    images = images.to(torch.float32)
    labels = labels.to(args.device)
    outputs = model(images)

    # Loss #
    loss = loss_fn(outputs, labels)
    loss = loss / args.batch_rate
    loss.backward()
    if (i+1)%args.batch_rate==0:
      opt.step()
      opt.zero_grad()
    
    # Stats #
    total_samples += images.shape[0]
    loss_sum += loss.cpu().data.item() * outputs.shape[0]
    correct_samples = torch.sum(classifier(outputs)==labels).cpu().data.item()
    total_correct_samples += correct_samples

    # Print Stats #
    acc = total_correct_samples / total_samples

    print(f'\r\tBatch [{i+1}/{len(data)}] Training: {acc:.2%}',end="")

  opt.step()
  opt.zero_grad()
  return acc



from .quant import dequant
def qtest(model, data, classifier, args):
  model.eval()
  total_samples = 0
  loss_sum = 0
  total_correct_samples = 0
  with torch.no_grad():
    for i, (images, labels) in enumerate(data):
      
      # Run Model #
      # images = images.to(args.device)
      images = images.to(torch.float32)
      # images = torch.quantize_per_tensor(images, scale=1, zero_point=0, dtype=torch.quint8)
      # labels = labels.to(args.device)
      outputs = model(images)
      outputs = dequant(outputs)

      # Stats #
      total_samples += images.shape[0]
      correct_samples = torch.sum(classifier(outputs)==labels).cpu().data.item()
      total_correct_samples += correct_samples

      # Print Stats #
      acc = total_correct_samples / total_samples

      print(f'\r\tBatch [{i+1}/{len(data)}] Validation: {acc:.2%}',end="")
  return acc
