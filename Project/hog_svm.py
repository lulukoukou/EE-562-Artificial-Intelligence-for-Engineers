########################################################################
# HIGHLIGHT: use skimage.feature for HOG feature extraction
# HIGHLIGHT: use sklearn,svm for SVM classifier

import numpy as np
from skimage import feature
from sklearn.svm import SVC, LinearSVC
import os
import pickle
import cv2

model_dir = '/data/chang/cifar-10/results/hog_svm'
# os.makedirs(model_dir)
train_img = np.load('/data/chang/cifar-10/data/training_data.npy')
train_x = []
for img in train_img:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(3, 3), block_norm='L2-Hys')
    train_x.append(x)
train_x = np.array(train_x)

train_y = np.load('/data/chang/cifar-10/data/training_label.npy')

# clf = SVC(decision_function_shape='ovc')
clf = LinearSVC()
clf.fit(train_x, train_y)
pickle.dump(clf, open(model_dir + "/model", "wb"))

# clf = pickle.load(model_dir + "/model", "rb")
test_img = np.load('/data/chang/cifar-10/data/test_data.npy')
test_x = []
for img in test_img:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
           cells_per_block=(3, 3), block_norm='L2-Hys')
    test_x.append(x)
test_x = np.array(test_x)

test_label = np.load('/data/chang/cifar-10/data/test_label.npy')#[:100]
pred = clf.predict(test_x)
eval = pred == test_label
accuracy = sum(eval)/test_label.shape[0]

print('accuracy: %.4f' % accuracy)