
import torch
from torch import nn
import torchvision
torch.manual_seed(42)
torch.cuda.manual_seed(42)
def model_creation(num_classes,device):
  """
    Creates an EfficientNet-B0 model with a modified classifier head.
    
    Args:
        num_classes: Number of output classes for the modified classifier
        device: The device on which the model wil work
        """
  weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
  auto_transforms=weights.transforms()
  model=torchvision.models.efficientnet_b0(weights=weights).to(device)
  for param in model.features.parameters():
      param.requires_grad=False
  model.classifier=nn.Sequential(
      nn.Dropout(p=0.2,inplace=True),
      nn.Linear(in_features=1280,out_features=num_classes)
  ).to(device)
  
  return auto_transforms,model
