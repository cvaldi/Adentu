import cv2
import numpy as np
import matplotlib.pyplot as pl
from skimage import morphology
from skimage.measure import regionprops
from sklearn.linear_model import RANSACRegressor
import os

imgroot = '/home/ivan/Documents/Adentu/DATA/Paneles/1'
imglist = [os.path.join(imgroot, img) for img in sorted(os.listdir(imgroot))]

ksize = (21,11)

for imgpath in imglist:
    
    #~ imgpath = '/home/ivan/Documents/Adentu/Paneles/1/DJI_0782.jpg'
    print imgpath
    I = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2GRAY)
    E = cv2.Canny(I, 10, 65)

    pl.imshow(I)
    pl.show()

    #~ pl.imshow(E, cmap='gray')
    #~ pl.show()

    EB = cv2.dilate(E,np.ones((5,5)))
    #EB_cleaned = morphology.remove_small_objects(morphology.label(255-EB, connectivity=2), min_size=64, connectivity=2).astype(np.uint8)
    #EB = cv2.threshold(EB_cleaned, 1, 255, cv2.THRESH_BINARY_INV)[1]
    ET = 255 - cv2.erode(EB,np.ones((4,4)))

    #~ pl.imshow(ET, cmap='gray')
    #~ pl.show()


    ET_cleaned = morphology.remove_small_objects(morphology.label(ET, connectivity=1), min_size=600, connectivity=1)
    ET_cleaned = (ET_cleaned>0).astype(np.uint8)*255

    #~ pl.imshow(ET_cleaned, cmap='gray')
    #~ pl.show()

    ET_cleaned2 = morphology.remove_small_objects(morphology.label(ET, connectivity=1), min_size=9000, connectivity=1)
    ET_cleaned2 = (ET_cleaned2>0).astype(np.uint8)*255
    ET_cleaned[ET_cleaned2>0] = 0
    ET2 = cv2.threshold(ET_cleaned, 1, 255, cv2.THRESH_BINARY)[1]

    #~ pl.imshow(ET2, cmap='gray')
    #~ pl.show()

    ET2_label = morphology.label(ET2, connectivity=1)

    regprops = regionprops(ET2_label)
    #~ print [reg.eccentricity for reg in regprops]
    #~ print [reg.area/reg.perimeter**2 for reg in regprops]

    FINAL = np.zeros(ET2.shape, np.uint8)
    for reg in regprops:
        if reg.area < 3000 and reg.solidity > 0.8 and 4*np.pi*reg.area/2*reg.perimeter**2 > 0.4 and reg.eccentricity < 0.95 and reg.eccentricity > 0.7:
            FINAL[ET2_label == reg.label] = 255

    pl.imshow(FINAL, cmap='gray')
    pl.show()

    FINAL_label = morphology.label(FINAL, connectivity=1)

    regpropsFINAL = regionprops(FINAL_label)
    print [reg.eccentricity for reg in regpropsFINAL]
    
    I2 = I.astype(float)
    I2[FINAL == 0] = 0

    #~ pl.imshow(I, cmap='gray')
    #~ pl.show()

    U,V = np.where(I2>0)

    X = np.array([U,V,U**2, V**2, U*V, np.ones(U.shape)]).T
    #~ X = np.array([U/640.,V/640.,U**2/640./640., V**2/640./640., U*V/640./640., np.ones(U.shape)]).T
    #~ X = np.array([U,V, np.ones(U.shape)]).T
    Y = I[U,V].ravel()

    model_ransac = RANSACRegressor()
    model_ransac.fit(X,Y)

    I2 = I.astype(float)
    U,V = np.where(I2>0)
    X = np.array([U,V,U**2, V**2, U*V, np.ones(U.shape)]).T
    #~ X = np.array([U/640.,V/640.,U**2/640./640., V**2/640./640., U*V/640./640., np.ones(U.shape)]).T
    #~ X = np.array([U,V, np.ones(U.shape)]).T
    Ypred = model_ransac.predict(X)
    I2[U,V] = I2[U,V]/Ypred

    #~ pl.imshow((I2>0.8)*(I2<1.3)*ET_cleaned, cmap='gray')
    pl.imshow(I2, vmin=0.8, vmax=1.4)
    pl.show()

#~ print model_ransac


#~ IM = np.zeros((I.shape[0]-ksize[0]+1, I.shape[1] - ksize[1]+1))

#~ for u in range(I.shape[0]-ksize[0]):
    #~ for v in range(I.shape[1]-ksize[1]):
        #~ IM[u,v] = np.std(I[u:u+ksize[0], v:v+ksize[1]])

#~ pl.imshow(IM)
#~ pl.show()
