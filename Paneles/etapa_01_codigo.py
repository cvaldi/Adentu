# coding: utf-8
from __future__ import print_function
from __future__ import division

# ###############################################################
# Codigo para etapa 01 de extracción de fallas en imágenes de
# paneles solares
#
# Versión 1.0
#
# Iván Lillo Vallés - Septiembre de 2018

# ###############################################################

import cv2
import os
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RANSACRegressor

import matplotlib.pyplot as pl
import pickle
import tifffile as tiff
import exifread 

class PanelSearch(object):
    '''
    Clase para realizar pruebas de la primera etapa de detección de fallas en paneles solares.
    
    Instanciar la clase:
    
        ps = PanelSearch()

    Funciones:
        ps.train_segmentation(masks_dir [, model_dir])
            Entrena un modelo para efectuar la segmentación de paneles a partir de máscaras
            anotadas en colores verde-rojo (cerde panel, rojo no panel). Guarda el modelo en
            la carpeta "models" relativa a donde se ejecuta el código, o se le puede pasar
            un parámetro opcional "model_dir" para especificar la ruta de los modelos.
            
        
    '''
    def __init__(self):
        self.models_dir = 'models'
        self.segm_model = None
        
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
        
        self.pts_origin = np.array(pts_origin).astype(float)
        self.pts_target = np.array(pts_target).astype(float)

        self.H = cv2.findHomography(self.pts_origin, self.pts_target)
        
        self.dx = 120
    
    def set_models_dir(self, mdir):
        if not os.path.exists(mdir):
            os.makedirs(mdir)
        self.models_dir = mdir
    
    ####### SEGMENTATION ############################################################################
    
    def __get_segm_feats(self, IRGB, ind_P = None):
        if ind_P is None:
            ind_P = np.where(IRGB[:,:,0] >= 0)
        feat_P = IRGB[ind_P].astype(np.float)
        mean_floor = IRGB[np.where(IRGB[:,:,2] < IRGB[:,:,0]+30)].mean(axis=0).reshape(1,-1)
        feat_P = np.concatenate((feat_P, np.tile(mean_floor, (len(feat_P),1)), np.ones((len(feat_P),1)) ), axis=1)
        return feat_P, ind_P
        
    def train_segmentation(self, masks_dir, model_dir = None, verbose=False):
        '''
        masks_dir: carpeta de imágenes, con las imágenes originales y sus máscaras
        model_dir (opcional): ruta de carpeta de modelos.
        '''
        if model_dir is not None:
            self.set_models_dir(model_dir)
        else:
            self.set_models_dir(self.models_dir)
        img_dir = masks_dir
        
        if verbose:
            print("Cargando imágenes desde " + masks_dir)
        # creacion de features

        FEAT_P = []
        FEAT_NP = []
        YP = []
        YNP = []
        for img_path in [os.path.join(img_dir, o) for o in os.listdir(img_dir) if 'JPG' in o]:
            if verbose:
                print(img_path)
            IRGB = cv2.cvtColor(cv2.imread(img_path.replace('_JPG','')), cv2.COLOR_BGR2RGB)
            IMASK = cv2.imread(img_path)

            ind_P = np.where(IMASK[:,:,1] > 200)
            ind_NP = np.where(IMASK[:,:,2] > 200)
            
            feat_P,_ = self.__get_segm_feats(IRGB, ind_P)
            feat_P = feat_P[::500]
            feat_NP,_ = self.__get_segm_feats(IRGB, ind_NP)
            feat_NP = feat_NP[::1000]

            FEAT_P.extend(feat_P.tolist())
            YP.extend([1]*len(feat_P))
            FEAT_NP.extend(feat_NP.tolist())
            YNP.extend([-1]*len(feat_NP))

        if verbose:
            print("Número de pixeles positivos: " + str(len(FEAT_P)))
            print("Número de pixeles negativos: " + str(len(FEAT_NP)))
        
        model = LinearSVC(class_weight='balanced')
        X = np.array(FEAT_P + FEAT_NP).astype(np.float)
        Y = np.array(YP + YNP)
        model.fit(X,Y)
        pickle.dump(model, open(os.path.join(self.models_dir, 'segm.pkl'), 'wb'))
        if verbose:
            print("Modelo de segmentación guardado en " + os.path.join(self.models_dir, 'segm.pkl'))
    
    def segmentation(self, img_path, show_img = False, out_path = None):
        if type(img_path) == str or type(img_path) == unicode:
            if not os.path.exists(img_path):
                print("Imagen " + img_path + " no existe!")
                return None
        if self.segm_model is None:
            if not os.path.exists(os.path.join(self.models_dir, 'segm.pkl')):
                print("Modelo de segmentación no ha sido creado!")
                return None
            self.segm_model = pickle.load(open(os.path.join(self.models_dir, 'segm.pkl'), 'rb'))
        if self.segm_model:
            if type(img_path) == str or type(img_path) == unicode:
                IRGB = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            else:
                IRGB = img_path
            
            feat_P, ind = self.__get_segm_feats(IRGB)
            
            clasif = self.segm_model.predict(feat_P)
            
            IBW = np.zeros(IRGB.shape[:2]).astype(np.uint8)
            IBW[ind] = ((clasif+1)/2*255).astype(np.uint8)
            
            IBW = cv2.dilate(cv2.erode(IBW, np.ones((10,10))), np.ones((10,10)))
            IBW = cv2.erode(cv2.dilate(IBW, np.ones((4,4))), np.ones((10,10)))
            
            IBW = IBW // 255
            
            #~ IBW_NOT = cv2.erode(cv2.dilate(IBW_NOT, np.ones((10,10))), np.ones((10,10)))
            #~ IBW = 255 - IBW_NOT
            
            ISEGM = cv2.cvtColor(IRGB*IBW[:,:,None], cv2.COLOR_RGB2BGR)
            
            if show_img:
                pl.imshow(ISEGM)
                pl.show()
            
            if out_path:
                if not os.path.exists(os.path.dirname(out_path)):
                    os.makedirs(os.path.dirname(out_path))
                cv2.imwrite(out_path, ISEGM)
            
            return ISEGM, IBW
    
    ####### THERMAL AND RGB IMAGE REGISTRATION #########################################################

    def get_exif_data(self, imrgb):
        return exifread.process_file(open(imrgb,'rb'))
        
    def register_thermal_RGB(self, imrgb, imtiff, min_height):
        if not os.path.exists(imrgb):
            print("Imagen RGB inexistente: " + imrgb)
            return None, None
        if not os.path.exists(imtiff):
            print("Imagen TIFF inexistente: " + imtiff)
            return None, None
        
        IRGB = cv2.cvtColor(cv2.imread(imrgb), cv2.COLOR_BGR2RGB)
        ITERM = tiff.imread(imtiff).astype(float)
        
        exif = self.get_exif_data(imrgb)
        
        height = eval(str(exif['GPS GPSAltitude']))
        
        print("Altitud: " + str(height))
        if height < min_height:
            print("Imagen descartada")
            return None, None
            
        ITERM2 = ((ITERM - ITERM.min())/(ITERM.max()-ITERM.min())*255).astype(np.uint8)
        
        I2 = cv2.warpPerspective(IRGB, self.H[0], ITERM2.shape[::-1], flags=cv2.INTER_AREA)
        
        ERGB = cv2.Canny(I2, 250, 350)
        ERGB[np.where(cv2.dilate((I2[:,:,0]==0).astype(np.uint8)*255, np.ones((10,10))))] = 0
        ERGB = ERGB[self.dx:-self.dx, self.dx:-self.dx]
        
        ERGB = cv2.dilate(ERGB, np.ones((5,5)))
        ETERM = cv2.Canny(ITERM2, 10, 140)
        ETERM = cv2.dilate(ETERM, np.ones((5,5)))

        res = cv2.matchTemplate(ERGB, ETERM, cv2.TM_CCOEFF_NORMED)
        a = res.argmax()

        xd = a%res.shape[0] - self.dx
        yd = a//res.shape[0] - self.dx

        H2 = np.array([[1,0,xd],[0,1,yd],[0,0,1]]).astype(np.float)
        I3 = cv2.warpPerspective(I2, H2, (I2.shape[1], I2.shape[0]), flags=cv2.INTER_AREA)
        
        ISEGM, IBW = self.segmentation(I3)

        return ITERM*IBW, IRGB

    def detect_hotspots(self, imrgb, imtiff, min_height = 0):
        print("Detectando puntos calientes en imagen " + imrgb)
        
        ITERM_reg, IRGB = self.register_thermal_RGB(imrgb, imtiff, min_height)
        
        idx = np.where(ITERM_reg>0)
        if len(idx[0]) == 0:
            return None, None
        vmean = ITERM_reg[np.where(ITERM_reg>0)].mean()
  
        ITERM = tiff.imread(imtiff).astype(float)
        ITERM = cv2.cvtColor(np.maximum(np.minimum((ITERM - vmean + 100)/(1.2), 255), 0).astype(np.uint8), cv2.COLOR_GRAY2RGB)
       
        im2, ctrs, hier = cv2.findContours((ITERM_reg > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        IPRED = np.zeros(ITERM_reg.shape)
        
        for ctr in ctrs:
            I2 = (0*ITERM_reg).astype(np.uint8)
            cv2.drawContours(I2, [ctr], -1, 255, -1)            
            I3 = cv2.erode(I2, np.ones((10,10)))
            idx = np.where(I3>0)
            idx_orig = np.where(I2 > 0)
            idx_contour = np.where((I3==0) & (I2>0))
            ITERM_reg[idx_contour] = 0
            if len(idx) and len(idx[0]) > 500: 
                X = np.concatenate( (np.array(idx).T[:,::-1], np.ones(len(idx[0])).reshape(-1,1)), axis=1)
                Y = ITERM_reg[idx]
                model = RANSACRegressor(residual_threshold = 20)
                model.fit(X,Y)
                pred = model.predict(X)
                IPRED[idx] = pred
                
            elif len(idx):
                ITERM_reg[idx] = 0
        
        IFAIL = ((ITERM_reg - IPRED) > 75).astype(np.uint8)
        IFAIL = cv2.dilate(IFAIL, np.ones((10,10)))

        im2, ctrs, hier = cv2.findContours(IFAIL, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        final_ctrs = []
        for ctr in ctrs:
            if cv2.contourArea(ctr) < 700:
                cv2.drawContours(ITERM,[ctr],0,(255,0,0),2)
                final_ctrs.append(ctr)
        
        return ITERM, final_ctrs
        

if __name__ == "__main__":
    ps = PanelSearch()
    img_dir = '/home/ivan/Documents/Adentu/DATA/Paneles/Vuelos/Vuelo_17_38m-7ms-SF72-SL80-GSD5_(1de4)/Imagenes'
    RGBlist = sorted([os.path.join(img_dir, o) for o in os.listdir(img_dir) if 'digital.jpg' in o])
    RAWlist = sorted([os.path.join(img_dir, o) for o in os.listdir(img_dir) if 'radiometric.tiff' in o])

    print(len(RGBlist))

    for imrgb, imtiff in zip(RGBlist, RAWlist):
        ITERM,_ = ps.detect_hotspots(imrgb, imtiff)
        pl.imshow(ITERM)
        pl.show()
