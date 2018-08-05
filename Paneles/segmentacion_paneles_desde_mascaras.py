# coding: utf-8
from __future__ import print_function
from __future__ import division

import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

img_dir = '/home/ivan/Documents/Adentu/Paneles/mascara_paneles'

# creacion de features

FEAT_P = []
FEAT_NP = []
YP = []
YNP = []

for img_path in [os.path.join(img_dir, o) for o in os.listdir(img_dir) if 'JPG' in o]:
    IRGB = cv2.imread(img_path.replace('_JPG',''))
    IMASK = cv2.imread(img_path)
    

    ind_P = np.where(IMASK[:,:,1] > 200)
    ind_NP = np.where(IMASK[:,:,2] > 200)

    feat_P = IRGB[ind_P][::50]
    feat_NP = IRGB[ind_NP][::200]

    FEAT_P.extend(feat_P.tolist())
    YP.extend([1]*len(feat_P))
    FEAT_NP.extend(feat_NP.tolist())
    YNP.extend([-1]*len(feat_NP))

print (len(FEAT_P))
print (len(FEAT_NP))

#~ model = SVC(kernel='linear')
model = DecisionTreeClassifier()
#~ model = RandomForestClassifier()

X = np.array(FEAT_P + FEAT_NP)
Y = np.array(YP + YNP)

print(X.shape)
print(Y.shape)

scores = cross_val_score(model, X, Y, cv=10)

print(scores)

model.fit(X,Y)




    
    
    

