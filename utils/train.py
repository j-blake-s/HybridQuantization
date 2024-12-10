

import torch
def test(model, data, classifier, args):
  model.eval()
  total_samples = 0
  total_correct_samples = 0
  for i, (images, labels) in enumerate(data):
    
    # Run Model #
    images = images.to(args.device)
    images = images.to(torch.float32)
    labels = labels.to(args.device)
    outputs = model(images)

    
    # Stats #
    total_samples += images.shape[0]
    correct_samples = torch.sum(classifier(outputs)==labels).cpu().data.item()
    total_correct_samples += correct_samples

    # Print Stats #
    acc = total_correct_samples / total_samples

    # print(f'\r\tBatch [{i+1}/{len(data)}] Training: {acc:.2%}',end="")

  return acc


