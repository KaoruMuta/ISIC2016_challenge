import os
from sklearn.cluster import KMeans
from skimage import data
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

totalfiles = []
imageFormats = ['.jpg', '.png', '.jpeg']
isic2016train = pd.read_csv('ISBI2016_ISIC_Part3_Training_GroundTruth.csv', header=None)
isic2016test = pd.read_csv('ISBI2016_ISIC_Part3_Test_GroundTruth.csv', header=None)
label = []
for root, dirs, files in os.walk('2016train'):
    for file in files:
        for imageFormat in imageFormats:
            if file.endswith(imageFormat):
                totalfiles.append(os.path.abspath(os.path.join(root, file)))
                break

for i in range(900):
    if file.split(os.sep)[-1] == isic2016train.iloc[i][0] + '.jpg':
        if isic2016train.iloc[i][1] == 'benign':
            label.append(0)
        else:
            label.append(1)
        break

print(len(totalfiles))
for i, image in enumerate(totalfiles):
    img = Image.open(image)
    img = img.convert('RGB')
    img = img.resize((256, 256))
    if label[i] == 0:
        img.save(os.path.join('dc/0', image.split(os.sep)[-1]))
    else:
        img.save(os.path.join('dc/1', image.split(os.sep)[-1]))

feature = np.array([data.imread(f'./dc/0/{path}') for path in os.listdir('clustering')])
print(feature.shape)
print(len(feature))
feature = feature.reshape(len(feature), -1).astype(np.float64)
plt.figure()
plt.scatter(feature[:, 0], feature[:, 1], c='blue', s=10, cmap='viridis')
plt.savefig('clustering1.png')
print(feature.shape)
print(len(feature))

#model = KMeans(n_clusters=3).fit(feature)
#labels = model.labels_
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(feature)
    y_kmeans = kmeans.predict(feature)

    plt.figure()
    plt.scatter(feature[:, 0], feature[:, 1], c=y_kmeans, s=10, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    plt.savefig('clustering' + str(i) + '.png')
