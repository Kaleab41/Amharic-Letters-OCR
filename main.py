from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
import os
from skimage.feature import hog
import joblib
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as pyplot

# train_dir = r'/content/drive/MyDrive/YourFolderName/'
train_dir = r'ocr_data/train/'
labels_list = [i for i in os.listdir(train_dir)]
pathimg = [os.listdir(train_dir + i) for i in labels_list]

# Visualize HOG for letter A
im_test = cv2.imread('ocr_data/train/1/0.jpg', 0)
_, hog_img = hog(im_test, orientations=9, pixels_per_cell=(
    8, 8), cells_per_block=(1, 1), visualize=True)
plt.imshow(hog_img, cmap='gray')
# cv2.imwrite('AHog.jpg',hog_img)

# extract the hog for each image and store it in a list with its
# corresponding label
features = []
labels = []
for i, j in enumerate(zip(pathimg, labels_list)):
    imgs, label = j
    for img in imgs:
        img = cv2.imread(train_dir+label+'/'+img)
        img_res = cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)
        img_gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        hog_img = hog(img_gray, orientations=9,
                      pixels_per_cell=(8, 8), cells_per_block=(1, 1))
        features.append(hog_img)
        labels.append(label)

print(len(pd.DataFrame(np.array(features))))
print(len(pd.DataFrame(np.array(labels))))

df = pd.DataFrame(np.array(features))
df['target'] = labels
df

df['target'].unique()

# df.target.value_counts()
sns.countplot(x='target', data=df)

"""#Training"""

x = np.array(df.iloc[:, :-1])
y = np.array(df['target'])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.90,
                                                    random_state=42)

# sm = SMOTE(random_state=0)
oversample = SMOTE(random_state=0)
sm_x, sm_y = oversample.fit_resample(x_train, y_train)

bal_df = pd.DataFrame(sm_x)
bal_df['target'] = pd.DataFrame(sm_y)
sns.countplot(x='target', data=bal_df)

bal_df['target'].value_counts()

lreg = LogisticRegression()
clf = lreg.fit(sm_x, sm_y)
y_pred = clf.predict(x_test)
print('Accuracy {:.2f}'.format(clf.score(x_test, y_test)))

print(classification_report(y_test, y_pred))

joblib.dump(clf, r'ocr_data/models/hog_trained.pkl')

clf = joblib.load('ocr_data/models/hog_trained.pkl')
im = cv2.imread('ocr_data/licenseplates/th.jpg')

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

ret, im_th = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

ctrs, hier = cv2.findContours(im_th, cv2.RETR_TREE,
                              cv2.CHAIN_APPROX_SIMPLE)
bboxes = [cv2.boundingRect(c) for c in ctrs]
sorted_bboxes = sorted(bboxes, key=lambda b: b[0])

plate_char = []

for num, i_bboxes in enumerate(sorted_bboxes):
    [x, y, w, h] = i_bboxes
    if h > 100 and w < 100:
        # Make the rectangular region around the digit
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1)
        roi = im_gray[y:y+h, x:x+w]
        # Resize the image
        roi = cv2.resize(roi, (64, 128), interpolation=cv2.INTER_AREA)
        # Calculate the HOG features
        # use the same parameters used for training
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(10, 10))
        nbr = clf.predict(np.array([roi_hog_fd]))

        cv2.putText(im, str((nbr[0])), (x, y+h), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 200, 250), 3)
        plate_char.append(str(nbr[0]))

print(''.join(plate_char))

# cv2.imshow('result', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# img = cv2.imread('path')
pyplot.imshow(im)
pyplot.show()
