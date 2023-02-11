import os
import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
import numpy as np
from tqdm import tqdm
import cv2
import os
import PIL
from glob import glob
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import tensorflow as tf
from skimage import measure
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.activations import *



def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def prepare_img_arr(df = pd.DataFrame(), resize_shape = tuple(), color_mode = "rgb"):
    im_arr = list()

    for im_path in tqdm(paths_df.im_path):
        resized_image = cv2.resize(cv2.imread(im_path),resize_shape)
        resized_image = resized_image/255.
        if color_mode == "gray":
            im_arr.append(resized_image[:,:,0])
        elif color_mode == "rgb":
            im_arr.append(resized_image[:,:,:])

    return im_arr



paths_for_lungs=glob("/content/drive/MyDrive/shinzen/lung_segmentation/CXR_png/*.png")

paths_related = defaultdict(list)

for path_of_img in paths_for_lungs:
    check_img_match = re.search("CXR_png/(.*)\.png$", path_of_img)
    if check_img_match:
        img_name = check_img_match.group(1)
        paths_related["im_path"].append(path_of_img)

a = pd.DataFrame.from_dict(paths_related)
im_arr = prepare_img_arr(df = paths_df, resize_shape = (256,256), color_mode = "gray")

im_arr = np.array(im_arr).reshape(len(im_arr), 256, 256, 1)


loaded_model = pickle.load(open("unet_model.pkl, 'rb')
predicted_masks = loaded_model.predict(im_arr)

for pred in predicted_masks:
    plt.imshow(pred)