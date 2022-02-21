import numpy as np
import glob
import cv2
import math
import pickle
from scipy.linalg import svd
from numpy import linalg as LA
from numpy import mean, std
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, plot_confusion_matrix

from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from cf_matrix import make_confusion_matrix
### Dataset with detected faces directory
path = glob.glob("D:\\FER_DESERTATION\\CODE\\DATASET\\*\*.jpeg")
X = []
count = 0
### Read each image resize it to 50,50 pixels for the models and append the resized array to a X List
for img in path:
    # if "RGB" in img:
    n = cv2.imread(img)
    gray = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (50, 50))
    # cv2.imshow('',gray)
    # cv2.waitKey(1)
    X.append(gray)
    count = count + 1
    # if count == 1000:
    #     break

#############################
# Create a list A with the dimensions 50, 50 and length of data (total images in our case)
sizeImg = X[0].shape
A = np.zeros((sizeImg[0] * sizeImg[1], len(X)))
# Flatten the array
for i in range(0, len(X)):
    tmp = (np.array(X[i]).reshape(-1))
    A[:, i] = np.array(tmp)
# Creating a labels list
Y = np.zeros((len(X)))
# Tagging each image label and updating Y accordingly
i = 0
for img in path:
    if "anger" in img:
        Y[i] = 0
    elif "disgust" in img:
        Y[i] = 1
    elif "fear" in img:
        Y[i] = 2
    elif "happy" in img:
        Y[i] = 3
    elif "neutral" in img:
        Y[i] = 4
    elif "suprise" in img:
        Y[i] = 5
    elif "sad" in img:
        Y[i] = 6
    i = i + 1

A = A.T
# Spliting the data into train and test, 80% train and 20% test
## giving random_state as 15 to shuffle the data accordingly
random_state = 12883823
train_X, test_X, train_Y, test_Y = train_test_split(A, Y, test_size=0.40, random_state=random_state)
# rkf = RepeatedKFold(random_state=random_state)
# Printing the shape of dataset we have
print(f'train_X.shape : {train_X.shape}')
print(f'test_X.shape : {test_X.shape}')
print(f'train_Y.shape : {train_Y.shape}')
print(f'test_Y.shape : {test_Y.shape}')
##############
# print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)
# Preprocessing the data for our model
sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)
###########################
# Creating a support vector classifier using SVC class of scikit learn
classifier_svm = SVC(kernel='linear', random_state=random_state, max_iter=100)
# Training the SVC by using fit method
classifier_svm.fit(train_X, train_Y)
## Saving the model to disk for future use
pickle.dump(classifier_svm, open('SVM_Model.sav', 'wb'))
####
# Creating a K-KNeighborsClassifier with n_neighbors set to 5 and metric as minkowski, moreover
# p = 2 as 2 inidicates euclidean distance and 1 inidicates mahaatn
classifier_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
# Training the classifier
classifier_knn.fit(train_X, train_Y)
#####
# Saving the KNN classifier to disk for future use
pickle.dump(classifier_knn, open('KNN_MODEL.sav', 'wb'))
################
# Creaing a plot to visualize the performance of both svc and knn
# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=random_state, shuffle=True)
##
pred_y = classifier_svm.predict(test_X)
scores = cross_val_score(classifier_svm, test_X, pred_y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('SVC cross-val Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
svc_conf_matrix = confusion_matrix(test_Y, pred_y)
make_confusion_matrix(svc_conf_matrix, ['anger', 'disgust', 'fear','happy','neutral','suprise','sad'],
                      ['anger', 'disgust', 'fear','happy','neutral','suprise','sad'],title="SVC ConfusionMatrix")
plt.show()
################
cv = KFold(n_splits=10, random_state=random_state, shuffle=True)
pred_y = classifier_knn.predict(test_X)
svc_conf_matrix_2 = confusion_matrix(test_Y, pred_y)
scores = cross_val_score(classifier_knn, test_X, pred_y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('KNN cross-val Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
make_confusion_matrix(svc_conf_matrix_2,['anger', 'disgust', 'fear','happy','neutral','suprise','sad'],
                      ['anger', 'disgust', 'fear','happy','neutral','suprise','sad'],title="KNN ConfusionMatrix")

plt.show()
#####
pred_Y = classifier_svm.predict(test_X)
test_Y = test_Y.tolist()
###
y_test_np = np.asarray(test_Y)
disc = y_test_np - pred_Y

count = 0
for i in disc:
    if i == 0:
        count += 1
accuracy = ((100 * count) / len(pred_Y))
# Accuracy of SVC
print(f'Support Vector Classification Accuracy {accuracy}')

correct = np.where(pred_Y == test_Y)[0]
# Number of correct labels found
print("SVC Found %d correct labels" % len(correct))
print(classification_report(test_Y, pred_Y, target_names=['anger', 'disgust', 'fear','happy','neutral','suprise','sad']))
#######
# tesing accuracy of KNN
pred_Y = classifier_knn.predict(test_X)
###
y_test_np = np.asarray(test_Y)
disc = y_test_np - pred_Y

count = 0
for i in disc:
    if i == 0:
        count += 1
accuracy = ((100 * count) / len(pred_Y))
# accuracy of knn
print(classification_report(test_Y, pred_Y, target_names=['anger', 'disgust', 'fear','happy','neutral','suprise','sad']))

print(f'KNN ACCURACY {accuracy}')
####
correct = np.where(pred_Y == test_Y)[0]
print("KNN Found %d correct labels" % len(correct))