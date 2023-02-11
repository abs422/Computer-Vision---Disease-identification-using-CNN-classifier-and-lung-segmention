# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:08:53 2022

@author: Abhishek Srivastava
"""

import cv2 as cv2
import numpy as np
from skimage import measure
from sys import platform as sys_pf
import warnings
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt

cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/thresh.png', thresh)

#%%
# Read the image as grayscale
img = cv2.imread("C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/archive (3)/data/Lung Segmentation/CXR_png/CHNCXR_0001_0.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
filtro = cv2.pyrMeanShiftFiltering(img, 0, 40)
gray = cv2.cvtColor(filtro, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

#%%
img = cv2.imread("C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/archive (3)/data/Lung Segmentation/CXR_png/CHNCXR_0001_0.png")
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray,180,255,cv2.THRESH_BINARY_INV)
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/thresh.png', thresh)

kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 1)
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/closing.png', closing)
sure_bg = cv2.dilate(closing,kernel,iterations=3)
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/sure_bg.png', sure_bg)

dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/dist_transform.png', dist_transform)

ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/sure_fg.png', sure_fg)
sure_fg = np.uint8(sure_fg)
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/sure_fg.png', sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/unknown.png', unknown)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+0
markers[unknown==255] = 255
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/markers.png', markers)

markers = cv2.watershed(img,markers)
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/markers.png', markers)

for i in range(0, markers.shape[0]):
    for j in range(0, markers.shape[1]):
        if markers[i][j] <10:
            markers[i][j] = 0
        if markers[i][j] >220:
            markers[i][j] = 256

seg_img = img
seg_img[markers == 0] = [0,0,255]
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/seg_img.png', seg_img)

ret, thresh_inv = cv2.threshold(gray,180,255,cv2.THRESH_BINARY)
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/thresh_inv.png', thresh_inv)

plt.subplot(211),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(img, 'gray')
plt.imsave(r'thresh.png',img)
plt.title("Watershed Segmented"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
            
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/markers.png', markers)

#%%
mask = cv2.imread("C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/archive (3)/Lung Segmentation/masks/CHNCXR_0001_0_mask.png")
mask_gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

for i in range(0, mask_gray.shape[0]):
    for j in range(0, mask_gray.shape[1]):
        if mask_gray[i][j] == 0:
            mask_gray[i][j] = 255
        elif mask_gray[i][j] >= 250:
            mask_gray[i][j] = 0

cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/mask_gray.png', mask_gray)
#%%

trial = markers + mask_gray

cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/trial.png', trial)
for i in range(0, trial.shape[0]):
    for j in range(0, trial.shape[1]):
        if trial[i][j] == 0:
            trial[i][j] = 255
        elif trial[i][j] >= 250:
            trial[i][j] = 0

opened = cv2.dilate(trial,kernel, iterations = 1)
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/trial.png', trial)
cv2.imwrite('C:/Lehigh/Study Material/DSCI 498 - Computer Vision/Project/Results/Image Outputs/openedtrial.png', opened)