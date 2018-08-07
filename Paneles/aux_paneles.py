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

IRGB = cv2.cvtColor(cv2.imread('/home/ivan/Documents/Adentu/DATA/Paneles/Vuelos/Vuelo_17_38m-7ms-SF72-SL80-GSD5_(1de4)/Imagenes/09-14-06-783_digital.jpg'), cv2.COLOR_BGR2RGB)

#pl.imshow(np.concatenate((IRGB[:,:,0], IRGB[:,:,1], IRGB[:,:,2]), axis=1), cmap='gray')
pl.imshow(IRGB[:,:,2] < IRGB[:,:,0]+30)
pl.show()
