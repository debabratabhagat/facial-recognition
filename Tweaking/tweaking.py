import pickle
import deepface
from imutils import paths
import os
import face_recognition
import pandas as pd
import numpy as np
import itertools
encodings = pickle.loads(open("../hog_encodings.pickle", "rb").read())

dataset= {}
for name in encodings['names']:
    dataset[name]=[]
# print(dataset)

for i,encoding in enumerate(encodings['encodings']):
    # print(i,":",encodings["names"][i])
    dataset[encodings["names"][i]].append(encoding)

print(len(dataset["debabrata"]))

positives = []
for key, values in dataset.items():
 for i in range(0, len(values)-1):
  for j in range(i+1, len(values)):
   positive = []
   positive.append(values[i])
   positive.append(values[j])
   positives.append(positive)
 
positives = pd.DataFrame(positives, columns = ["file_x", "file_y"])
positives["decision"] = "Yes"

# print(positives)

samples_list = list(dataset.values())
# print(len(dataset))
negatives = []
for i in range(0, len(dataset) - 1):
 for j in range(i+1, len(dataset)):
  cross_product = itertools.product(samples_list[i], samples_list[j])
  cross_product = list(cross_product)
 
  for cross_sample in cross_product:
   negative = []
   negative.append(cross_sample[0])
   negative.append(cross_sample[1])
   negatives.append(negative)
 
negatives = pd.DataFrame(negatives, columns = ["file_x", "file_y"])
negatives["decision"] = "No"
# print(positives)  

df = pd.concat([positives, negatives]).reset_index(drop = True)
 
df.file_x = df.file_x
df.file_y = df.file_y

instances = df[["file_x", "file_y","decision"]].values.tolist()
print(len(instances))
# print(instances[1000][2])
distances=[]
for instance in instances:
    a1 = np.array(instance[0]).reshape(1,-1)
    distance = face_recognition.face_distance(a1,instance[1])
    distances.append(distance[0])

df['distance'] = distances
df.to_csv("distance_lfw_hog.csv") 