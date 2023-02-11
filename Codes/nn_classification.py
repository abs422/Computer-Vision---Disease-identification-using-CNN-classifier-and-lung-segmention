# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 22:45:17 2022

@author: Abhishek Srivastava
"""

# Feature extraction and classfication
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

from glob import glob
from collections import defaultdict
import re

#%%

# getting paths
final_masks_paths=glob('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Final pred mask/*.png')

# Saving the classifier ready inputs in local 
for masks in final_masks_paths:
        
    fm = cv2.imread(masks)
    fm_gray = cv2.cvtColor(fm,cv2.COLOR_BGR2GRAY)
    ids = masks[-8:][:4]
    if int(masks[-8:][:4]) >= 327:
        org_img = cv2.imread(f'C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/archive (3)/Lung Segmentation/CXR_png/CHNCXR_{ids}_1.png')
    else:
        org_img = cv2.imread(f'C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/archive (3)/Lung Segmentation/CXR_png/CHNCXR_{ids}_0.png')
    org_img_gray = cv2.cvtColor(org_img,cv2.COLOR_BGR2GRAY)
    for i in range(0, fm_gray.shape[0]):
        for j in range(0, fm_gray.shape[1]):
            if fm_gray[i][j] == 0:
                fm_gray[i][j] = 255
            elif fm_gray[i][j] >= 250:
                fm_gray[i][j] = 0
                
    fp = org_img_gray + fm_gray
    for i in range(0, fm_gray.shape[0]):
        for j in range(0, fm_gray.shape[1]):
            if org_img_gray[i][j] + fm_gray[i][j] >= 250:
                org_img_gray[i][j] = 255
            else:
                org_img_gray[i][j] = org_img_gray[i][j]
    
    
    # cv2.imwrite(f'C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Input to feature map model/trial2.png', org_img_gray)
    cv2.imwrite(f'C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Input to feature map model/fm_img_{ids}.png', org_img_gray)