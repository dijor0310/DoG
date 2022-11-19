import pandas as pd
import os
import shutil

training_dir = ".\\content\\data\\training"
validation_dir = ".\\content\\data\\validation"

train_labels = os.path.join(training_dir, "train\\labels.csv")
train_labels_df = pd.read_csv(train_labels)
#print(train_labels_df)
print(train_labels_df.shape[0])

val_labels = os.path.join(validation_dir, "val\\labels.csv")
val_labels_df = pd.read_csv(val_labels)
#print(val_labels_df)
print(val_labels_df.shape[0])

#os.mkdir(os.path.join(training_dir, "highway"))
#os.mkdir(os.path.join(training_dir, "pathway"))

#os.mkdir(os.path.join(validation_dir, "highway"))
#os.mkdir(os.path.join(validation_dir, "pathway"))


for i in range(train_labels_df.shape[0]):
    if train_labels_df.loc[i]['highway']=='footway':
        path = os.path.join(training_dir, "train\\" + str(train_labels_df.loc[i]['image_id']) + ".jpg")
    try:
        shutil.move(path, os.path.join(training_dir, "footway"))
    except:
        pass
       
for i in range(train_labels_df.shape[0]):
    if train_labels_df.loc[i]['highway']=='primary':
        path = os.path.join(training_dir, "train\\" + str(train_labels_df.loc[i]['image_id']) + ".jpg")
    try:
        shutil.move(path, os.path.join(training_dir, "primary"))
    except:
        pass
for i in range(val_labels_df.shape[0]):
    if val_labels_df.loc[i]['highway']=='footway':
        path = os.path.join(validation_dir, "val\\" + str(val_labels_df.loc[i]['image_id']) + ".jpg")
    try:
        shutil.move(path, os.path.join(validation_dir, "footway"))
    except:
        pass
for i in range(val_labels_df.shape[0]):
    if val_labels_df.loc[i]['highway']=='primary':
        path = os.path.join(validation_dir, "val\\" + str(val_labels_df.loc[i]['image_id']) + ".jpg")
    try:
        shutil.move(path, os.path.join(validation_dir, "primary"))
    except:
        pass