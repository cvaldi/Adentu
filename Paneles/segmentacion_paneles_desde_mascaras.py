# coding: utf-8
from __future__ import print_function
from __future__ import division

import cv2
import os
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as pl


img_dir = '/home/ivan/Documents/Adentu/Paneles/mascara_paneles'

# creacion de features

FEAT_P = []
FEAT_NP = []
YP = []
YNP = []

for img_path in [os.path.join(img_dir, o) for o in os.listdir(img_dir) if 'JPG' in o]:
    IRGB = cv2.cvtColor(cv2.imread(img_path.replace('_JPG','')), cv2.COLOR_BGR2RGB)
    IMASK = cv2.imread(img_path)
    #ILAB = cv2.cvtColor(IRGB, cv2.COLOR_BGR2LAB)

    ind_P = np.where(IMASK[:,:,1] > 200)
    ind_NP = np.where(IMASK[:,:,2] > 200)
    
    mean_floor = IRGB[np.where(IRGB[:,:,2] < IRGB[:,:,0]+30)].mean(axis=0).reshape(1,-1)

    #~ feat_P = np.concatenate((IRGB[ind_P][::50], ILAB[ind_P][::50]), axis=1)
    #~ feat_NP = np.concatenate((IRGB[ind_NP][::200], ILAB[ind_NP][::200]), axis=1)
    feat_P = IRGB[ind_P][::500].astype(np.float)
    feat_P = np.concatenate((feat_P, np.tile(mean_floor, (len(feat_P),1))), axis=1)
    
    #~ feat_P = np.concatenate((feat_P, feat_P/255**2, (feat_P[:,0]*feat_P[:,1]/255).reshape(-1,1), (feat_P[:,0]*feat_P[:,2]/255).reshape(-1,1), (feat_P[:,1]*feat_P[:,2]/255).reshape(-1,1) ), axis = 1)
    feat_NP = IRGB[ind_NP][::2000].astype(np.float)
    feat_NP = np.concatenate((feat_NP, np.tile(mean_floor, (len(feat_NP),1))), axis=1)
    #~ feat_NP = np.concatenate((feat_NP, feat_NP/255**2, (feat_NP[:,0]*feat_NP[:,1]/255).reshape(-1,1), (feat_NP[:,0]*feat_NP[:,2]/255).reshape(-1,1), (feat_NP[:,1]*feat_NP[:,2]/255).reshape(-1,1) ), axis = 1)

    FEAT_P.extend(feat_P.tolist())
    YP.extend([1]*len(feat_P))
    FEAT_NP.extend(feat_NP.tolist())
    YNP.extend([-1]*len(feat_NP))

print (len(FEAT_P))
print (len(FEAT_NP))

model = LinearSVC(class_weight='balanced')
#~ model = DecisionTreeClassifier()
#~ model = RandomForestClassifier(n_estimators=8, max_depth=5)

X = np.array(FEAT_P + FEAT_NP).astype(np.float)
print (X.shape)
X = np.concatenate((X, np.ones((len(X),1))), axis=1)
Y = np.array(YP + YNP)

print(X.shape)
print(Y.shape)

scores = cross_val_score(model, X, Y, cv=10)

print(scores)

model.fit(X,Y)

# testing images


path_test = '/home/ivan/Documents/Adentu/DATA/Paneles/Vuelos/Vuelo_17_38m-7ms-SF72-SL80-GSD5_(1de4)/Imagenes'

img_test  = [os.path.join(path_test, o) for o in os.listdir(path_test) if 'digital' in o]

out_folder = 'output'

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
    
for img in img_test:
    IRGB = cv2.resize(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB), (320, 240))
    mean_floor = IRGB[np.where(IRGB[:,:,2] < IRGB[:,:,0]+30)].mean(axis=0).reshape(1,-1)
    
    ind = np.where(IRGB[:,:,0] >= 0)
    
    feat_P = IRGB[ind].astype(np.float)
    feat_P = np.concatenate((feat_P, np.tile(mean_floor, (len(feat_P),1))), axis=1)
    feat_P = np.concatenate((feat_P, np.ones((len(feat_P),1))), axis=1)
    #~ feat_P = np.concatenate((feat_P, feat_P/255**2, (feat_P[:,0]*feat_P[:,1]/255).reshape(-1,1), (feat_P[:,0]*feat_P[:,2]/255).reshape(-1,1), (feat_P[:,1]*feat_P[:,2]/255).reshape(-1,1) ), axis = 1)
    clasif = model.predict(feat_P)
    IBW = np.zeros(IRGB.shape[:2]).astype(np.uint8)
    IBW[ind] = ((clasif+1)/2*255).astype(np.uint8)
    
    #~ IBW = cv2.dilate(cv2.erode(IBW, np.ones((10,10))), np.ones((10,10)))
    #~ IBW = cv2.erode(cv2.dilate(IBW, np.ones((4,4))), np.ones((4,4)))
    
    IBW = IBW // 255
    
    #~ IBW_NOT = cv2.erode(cv2.dilate(IBW_NOT, np.ones((10,10))), np.ones((10,10)))
    #~ IBW = 255 - IBW_NOT
    
    cv2.imwrite(os.path.join(out_folder, os.path.basename(img)), cv2.cvtColor(IRGB*IBW[:,:,None], cv2.COLOR_RGB2BGR))
    #~ pl.figure()
    #~ pl.imshow(IRGB)
    #~ pl.figure()
    #~ pl.imshow(IBW)
    #~ pl.show()
