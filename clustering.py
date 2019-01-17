import os
from sklearn.cluster import KMeans
from skimage import data
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

totalfiles = []
imageFormats = ['.jpg', '.png', '.jpeg']
for root, dirs, files in os.walk('2016train'):
    for file in files:
        for imageFormat in imageFormats:
            if file.endswith(imageFormat):
                totalfiles.append(os.path.abspath(os.path.join(root, file)))
                break

print(len(totalfiles))
'''for image in totalfiles:
	img = Image.open(image)
	img = img.convert('RGB')
	img = img.resize((224, 224))
	img.save(os.path.join('clustering', image.split(os.sep)[-1]))'''

feature = np.array([data.imread(f'./clustering/{path}') for path in os.listdir('clustering')])
print(feature.shape)
print(len(feature))
feature = feature.reshape(len(feature), -1).astype(np.float64)
print(feature.shape)
print(len(feature))

#model = KMeans(n_clusters=3).fit(feature)
#labels = model.labels_

kmeans = KMeans(n_clusters=10)
kmeans.fit(feature)
y_kmeans = kmeans.predict(feature)

plt.figure()
plt.scatter(feature[:, 0], feature[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.savefig(os.path.join('clustering', 'clustering.png'))
