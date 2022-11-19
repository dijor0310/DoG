import requests
import cv2
import json
url = 'http://localhost:5000/api'
content_type = 'image/jpeg'

headers = {'content-type': content_type}

#files = {'media': open('165180662274601.jpg', 'rb')}
image = cv2.imread('165180662274601.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
_, img_encoded = cv2.imencode('.jpg', image)
# send http request with image and receive response
response = requests.post(url, data=img_encoded.tostring(), headers=headers)
print(json.loads(response.text))


#requests.post(url, files=files)
