# import cv2
# import numpy as np

# digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)
# test_digits = cv2.imread("test_digits.png", cv2.IMREAD_GRAYSCALE)

# rows = np.vsplit(digits, 50)
# cells = []
# for row in rows:
#     row_cells = np.hsplit(row, 50)
#     for cell in row_cells:
#         cell = cell.flatten()
#         cells.append(cell)
# cells = np.array(cells, dtype=np.float32)

# k = np.arange(10)
# print(k)
# cells_labels = np.repeat(k, 250)

# test_digits = np.vsplit(test_digits, 50)
# test_cells = []
# for d in test_digits:
#     d = d.flatten()
#     test_cells.append(d)

# test_cells = np.array(test_cells, dtype=np.float32)

# knn = cv2.ml.KNearest_create()
# knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
# ret, result, neigbhours, dist = knn.findNearest(test_cells, k=1)

# print(result)

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
import cv2
import csv
import glob

header = ['label']
for i in range(0, 784):
    header.append('pixel'+str(i))
with open('dataset.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(header)
for label in range(10):
    dirList = glob.glob("Training Data/"+str(label)+"/"+str(label)+"/*.jpg")

    for img_path in dirList:
        im = cv2.imread(img_path)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
        roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

        data = []
        data.append(label)
        rows, cols = roi.shape

        for i in range(rows):
            for j in range(cols):
                k = roi[i, j]
                if k > 100:
                    k = 1
                else:
                    k = 0
                data.append(k)
        with open('dataset.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)
# Load data set

data = pd.read_csv('dataset.csv')
data = shuffle(data)
# print(data)

# separate dependant and independant
X = data.drop(["label"], axis=1)
Y = data["label"]

# preview sample

idx = 314
img = X.loc[idx].values.reshape(28, 28)
print(Y[idx])
plt.imshow(img)
