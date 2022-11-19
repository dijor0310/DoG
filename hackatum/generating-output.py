import pandas as pd
import os
data_dir = ".\\content\\data\\issues"

import torch
import cv2
import torchvision.transforms as transforms
import argparse
from model import CNNModel

# the computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# list containing all the class labels
labels = [
    'footway', 'primary'
    ]

# initialize the model and load the trained weights
model = CNNModel().to(device)
checkpoint = torch.load('outputs/model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# define preprocess transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((2024,2024)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])  


dir = os.path.join(data_dir, "issues.csv")
df = pd.read_csv(dir)
output_df = pd.DataFrame(columns = ['image_id', 'initial', 'corrected'])
for i in range(df.shape[0]):
    image_id = df.loc[i]['image_id']
    # read and preprocess the image
    image = cv2.imread(os.path.join(data_dir, str(image_id) + ".jpg"))
    # convert to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image.to(device))
    output_label = torch.topk(outputs, 1)
    pred_class = labels[int(output_label.indices)]
    #print(pred_class)
    #output_df.append({'image_id' : image_id, 'initial' : df.loc[i]['highway'], 'corrected' : str(pred_class)}, ignore_index=True)
    df_tmp = {'image_id' : image_id, 'initial' : df.loc[i]['highway'], 'corrected' : str(pred_class)}
    output_df = output_df.append(df_tmp, ignore_index = True)
    #print(output_df)
    
print(output_df)
from pathlib import Path  
filepath = Path('.\\outputs\\outputs.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
output_df.to_csv("outputs.csv")
#print(df)