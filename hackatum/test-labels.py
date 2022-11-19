import pandas as pd
import os 

data_dir = ".\\content\\data"
validation_dir = ".\\content\\data\\validation"

val_labels = os.path.join(data_dir, "val\\labels.csv")
df = pd.read_csv(val_labels)


for i in range(df.shape[0]):
    if df.loc[i]["image_id"] == 826651648262806:
        print(df.loc[i]["highway"])