# Import libraries
import torch
import cv2
import torchvision.transforms as transforms
import argparse
from model import CNNModel
import pandas as pd

import numpy as np
from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)


# the computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# list containing all the class labels
labels = [
    'footway', 'primary'
    ]
model = CNNModel().to(device)
checkpoint = torch.load('..\\hackatum\\outputs\\model.pth', map_location=device)
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


# Load the model
#model = pickle.load(open('model.pkl','rb'))
@app.route('/api',methods=['POST'])
def predict():
    #print("test")
    # Get the data from the POST request.
    #data = request.get_json(force=True)
    #r = requests.get(settings.STATICMAP_URL.format(**data), stream=True)
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Make prediction using model loaded from disk as per the data.
    #print(r)
    #print("test1")
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    print("test2")
    with torch.no_grad():
        outputs = model(image.to(device))
    output_label = torch.topk(outputs, 1)
    pred_class = labels[int(output_label.indices)]
    print(pred_class)
    return jsonify(pred_class)
if __name__ == '__main__':
    app.run(port=5000, debug=True)