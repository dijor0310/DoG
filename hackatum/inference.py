import torch
import torchvision
from torchvision import datasets, transforms

traindir = '.\content\data\\training'
testdir = '.\content\data\\validation'

#transformations
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      torchvision.transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],
    ),
                                      ])

#datasets
#train_data = datasets.ImageFolder(traindir,transform=train_transforms)
test_data = datasets.ImageFolder(testdir,transform=test_transforms)


import numpy as np

import matplotlib.pyplot as plt
def inference(test_data):
  idx = torch.randint(1, len(test_data), (1,))
  sample = torch.unsqueeze(test_data[122][0], dim=0).to(device)

  if torch.sigmoid(model(sample)) < 0.5:
    print("Prediction : Cat")
  else:
    print("Prediction : Dog")

  np_arr = test_data[122][0].permute(1, 2, 0).cpu().detach().numpy()

  plt.imshow(np_arr)
  plt.show()

import numpy as np
from torchvision import datasets, models, transforms
import torch.nn as nn

model = models.resnet18(pretrained=True)
for params in model.parameters():
  params.requires_grad_ = False

#add a new final layer
nr_filters = model.fc.in_features  #number of input features of last layer
model.fc = nn.Linear(nr_filters, 1)

device = "cpu"
model = model.to(device)
model.load_state_dict(torch.load(".\\highway_classifier.pt"))

inference(test_data)
