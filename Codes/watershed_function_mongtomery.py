# -*- coding: utf-8 -*-
"""
Created on Wed Nov 07 01:32:45 2022

@author: Abhishek Srivastava
"""
# Doing same segmentation pipeline for mongtomery datasets
import cv2 as cv2
import numpy as np
import pandas as pd
from skimage import measure
from sys import platform as sys_pf
import warnings
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.activations import *

#%%
identifier = ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010', '0011', '0012', '0013', '0014', '0015'
, '0016', '0017', '0018', '0019', '0020', '0021', '0022', '0023', '0024', '0025', '0026', '0027', '0028', '0029', '0030', '0031', '0032', '0033'
, '0034', '0035', '0036', '0037', '0038', '0039', '0040', '0041', '0042', '0043', '0044', '0045', '0046', '0047', '0048', '0049', '0050'
, '0051', '0052', '0053', '0054', '0055', '0056', '0057', '0058', '0059', '0060', '0061', '0062', '0063', '0064', '0065', '0066', '0067'
, '0068', '0069', '0070', '0071', '0072', '0073', '0074', '0075', '0076', '0077', '0078', '0079', '0080', '0081', '0082', '0083', '0084'
, '0085', '0086', '0087', '0088', '0089', '0090', '0091', '0092', '0093', '0094', '0095', '0096', '0097', '0098', '0099', '0100', '0101'
, '0102', '0103','0104'
, '0108'
, '0113'
, '0117'
, '0126'
, '0140'
, '0141'
, '0142'
, '0144'
, '0150'
, '0162'
, '0166'
, '0170'
, '0173'
, '0182'
, '0188'
, '0194'
, '0195'
, '0196'
, '0203'
, '0213'
, '0218'
, '0223'
, '0228'
, '0243'
, '0251'
, '0253'
, '0254'
, '0255'
, '0258'
, '0264'
, '0266'
, '0275'
, '0282'
, '0289'
, '0294'
, '0301'
, '0309'
, '0311'
, '0313'
, '0316'
, '0331'
, '0334'
, '0338'
, '0348'
, '0350'
, '0352'
, '0354'
, '0362'
, '0367'
, '0369'
, '0372'
, '0375'
, '0383'
, '0387'
, '0390'
, '0393'
, '0399']
#%%
from glob import glob
from collections import defaultdict
import re

#%%
def watershed_morpho(identifier):
    pred_mask = [[0]]
    try:
        img = cv2.imread(f'C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/archive (3)/Lung Segmentation/CXR_png/MCUCXR_{identifier}_1.png')
        mask = cv2.imread(f'C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/archive (3)/Lung Segmentation/masks/MCUCXR_{identifier}_1.png')
        
        b,g,r = cv2.split(img)
        rgb_img = cv2.merge([r,g,b])
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(gray,180,255,cv2.THRESH_BINARY_INV)
        
        
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 1)
        
        sure_bg = cv2.dilate(closing,kernel,iterations=3)
        
        
        dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
        
        
        ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
        
        sure_fg = np.uint8(sure_fg)
        
        unknown = cv2.subtract(sure_bg,sure_fg)
        
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+0
        markers[unknown==255] = 255
        
        
        markers = cv2.watershed(img,markers)
        
        
        for i in range(0, markers.shape[0]):
            for j in range(0, markers.shape[1]):
                if markers[i][j] <10:
                    markers[i][j] = 0
                if markers[i][j] >220:
                    markers[i][j] = 256
        
        seg_img = img
        seg_img[markers == 0] = [0,0,255]
        
        
        # ret, thresh_inv = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
        # cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/thresh_inv.png', thresh_inv)
        
        # plt.subplot(211),plt.imshow(rgb_img)
        # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(212),plt.imshow(img, 'gray')
        # plt.imsave(r'thresh.png',img)
        # plt.title("Watershed Segmented"), plt.xticks([]), plt.yticks([])
        # plt.tight_layout()
        # plt.show()
        
        
        mask_gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        
        for i in range(0, mask_gray.shape[0]):
            for j in range(0, mask_gray.shape[1]):
                if mask_gray[i][j] == 0:
                    mask_gray[i][j] = 255
                elif mask_gray[i][j] >= 250:
                    mask_gray[i][j] = 0
        
        pred_mask = markers + mask_gray
        
        for i in range(0, pred_mask.shape[0]):
            for j in range(0, pred_mask.shape[1]):
                if pred_mask[i][j] == 0:
                    pred_mask[i][j] = 255
                elif pred_mask[i][j] >= 250:
                    pred_mask[i][j] = 0
    except: 
        pass
    # fin_pred_mask = cv2.dilate(pred_mask,kernel,iterations=3)
    return cv2.imwrite(f'C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Pred_mask_watershed/pred_mask_mcucxr{identifier}.png', pred_mask)

#%%


for i in range(105,len(identifier)):
    watershed_morpho(identifier[i])

#%%
kernel = np.ones((5,5),np.uint8)

pred_masks_paths=glob('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Pred_mask_watershed_mong/*.png')

for masks in pred_masks_paths:
    ids = masks[-8:][:4]
    pm = cv2.imread(masks)
    pm_gray = cv2.cvtColor(pm,cv2.COLOR_BGR2GRAY)
    
    for i in range(0, pm_gray.shape[0]):
        for j in range(0, pm_gray.shape[1]):
            if pm_gray[i][j] == 0:
                pm_gray[i][j] = 0
            else:
                pm_gray[i][j] = 255
    
    fin_pred_mask = cv2.dilate(pm_gray,kernel,iterations=15)
    
    cv2.imwrite(f'C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Final pred mask mong/fin_pred_mask_mong{ids}.png', fin_pred_mask)


#%%

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2 * intersection) / (keras.sum(y_true_f) + keras.sum(y_pred_f))

def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

final_masks_paths=glob('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Final pred mask mong/*.png')
wat_masks_paths = glob('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Pred_mask_watershed_mong/*.png')

# Lets create empty list to store dice (IOU) score for each image
dc1 = []
dc2 = []
y_true_list = []
y_pred_list = []

for masks in final_masks_paths:    
        y_pred = cv2.imread(masks)
        ids = masks[-8:][:4]
        if int(masks[-8:][:4]) >= 104:
            y_true = cv2.imread(f'C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/archive (3)/Lung Segmentation/masks/MCUCXR_{ids}_1.png')
        else:
            y_true = cv2.imread(f'C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/archive (3)/Lung Segmentation/masks/MCUCXR_{ids}_0.png')
        
        y_true_gray = cv2.cvtColor(y_true,cv2.COLOR_BGR2GRAY)
        y_pred_gray = cv2.cvtColor(y_pred,cv2.COLOR_BGR2GRAY)
        
        dc2 = dc2 + [dice(y_pred_gray, y_true_gray, 255)]
        
        y_true_gray = cv2.resize(y_true_gray, (256, 256))
        y_pred_gray = cv2.resize(y_pred_gray, (256, 256))
        
        y_true_norm = y_true_gray/255
        y_pred_norm = y_pred_gray/255
        
        y_true_flatten = np.ndarray.flatten(y_true_norm)
        y_pred_flatten = np.ndarray.flatten(y_pred_norm)
        
        y_true_list = y_true_list + list(y_true_flatten)
        y_pred_list = y_pred_list + list(y_pred_flatten)
        
        # dc1 = dc1 + [dice_coef(y_true_gray, y_pred_gray)]
        
        


np.mean(dc2)
#0.905

#%%

for i in range(0,len(y_true_list)):
    if y_true_list[i] == 0.0:
        y_true_list[i] = 'Neg'
    else:
        y_true_list[i] = 'Pos'

for i in range(0,len(y_pred_list)):
    if y_pred_list[i] == 0.0:
        y_pred_list[i] = 'Neg'
    else:
        y_pred_list[i] = 'Pos'
#%%
from sklearn import metrics
# Printing the confusion matrix
# The columns will show the instances predicted for each label,
# and the rows will show the actual number of instances for each label.
print(metrics.confusion_matrix(y_true_list, y_pred_list, labels=["Neg", "Pos"]))
# Printing the precision and recall, among other metrics
print(metrics.classification_report(y_true_list, y_pred_list, labels=["Neg", "Pos"]))