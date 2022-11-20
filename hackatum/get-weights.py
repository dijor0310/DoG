import torch
import csv

model = torch.load('.\\outputs\\model.pth')
#print(model)

## open file for writing, "w" is writing
w = csv.writer(open(".\\outputs\\weights.csv", "w"))

# loop over dictionary keys and values
for key, val in model.items():

    # write every key and value to file
    w.writerow([key, val])
