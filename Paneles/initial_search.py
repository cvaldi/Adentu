import cv2
import tifffile as tiff
import exifread 
import matplotlib.pyplot as pl
import numpy as np


# ejemplo para obtener homografía
IRGB = cv2.cvtColor(cv2.imread('07-41-12-871_digital.jpg'), cv2.COLOR_BGR2RGB)
ITERM = tiff.imread('07-41-12-873_radiometric.tiff')

pts_origin = [  [279,30],
                [409,33],
                [130,750],
                [246,796],
                [649,110],
                [798,117],
                [498,832],
                [632,873],
                [1015,193],
                [1182,202],
                [860,913],
                [1013,959]]

pts_target = [  [175,66],
                [216,67],
                [125,298],
                [161,311],
                [292,92],
                [339,94],
                [241,324],
                [284,338],
                [408,118],
                [461,123],
                [358,351],
                [405,365]]
                
pts_origin = np.array(pts_origin).astype(float)
pts_target = np.array(pts_target).astype(float)

H = cv2.findHomography(pts_origin, pts_target)

I2 = cv2.warpPerspective(IRGB, H[0], ITERM.shape[::-1])
