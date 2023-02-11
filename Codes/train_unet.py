import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
from google.colab import drive
drive.mount('/content/drive/')
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

def unet(input_size=(256,256,1)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])
def prepare_train_test(df = pd.DataFrame(), resize_shape = tuple(), color_mode = "rgb"):
    im_arr = list()
    mask_arr = list()

    for im_path in tqdm(paths_df.im_path):
        resized_image = cv2.resize(cv2.imread(im_path),resize_shape)
        resized_image = resized_image/255.
        if color_mode == "gray":
            im_arr.append(resized_image[:,:,0])
        elif color_mode == "rgb":
            im_arr.append(resized_image[:,:,:])
  
    for path_of_mask in tqdm(paths_df.path_of_mask):
        resized_mask = cv2.resize(cv2.imread(path_of_mask),resize_shape)
        resized_mask = resized_mask/255.
        mask_arr.append(resized_mask[:,:,0])

    return im_arr, mask_arr




paths_for_lungs=glob("/content/drive/MyDrive/shinzen/lung_segmentation/CXR_png/*.png")
paths_for_masks=glob("/content/drive/MyDrive/shinzen/lung_segmentation/masks/*.png")

paths_related = defaultdict(list)

for path_of_img in paths_for_lungs:
    check_img_match = re.search("CXR_png/(.*)\.png$", path_of_img)
    if check_img_match:
        img_name = check_img_match.group(1)
    for path_of_mask in paths_for_masks:
        mask_match = re.search(img_name, path_of_mask)
        if mask_match:
            paths_related["im_path"].append(path_of_img)
            paths_related["path_of_mask"].append(path_of_mask)

a = pd.DataFrame.from_dict(paths_related)
im_arr, mask_arr = prepare_train_test(df = paths_df, resize_shape = (256,256), color_mode = "gray")


im_train, im_test, mask_train, mask_test = train_test_split(im_arr, mask_arr, test_size = 0.2, random_state= 42)


im_train = np.array(im_train).reshape(len(im_train), 256, 256, 1)
im_test = np.array(im_test).reshape(len(im_test), 256, 256, 1)
mask_train = np.array(mask_train).reshape(len(mask_train), 256, 256, 1)
mask_test = np.array(mask_test).reshape(len(mask_test), 256, 256, 1)





model = unet(input_size=(256,256,1))
model.compile(optimizer=Adam(lr=5*1e-4), loss="binary_crossentropy", 
                  metrics=[dice_coef, 'binary_accuracy'])
model.summary()

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
history = model.fit(x = im_train, 
                    y = mask_train, 
                    validation_data = (im_test, mask_test), 
                    epochs = 50, 
                    batch_size = 16,
                   callbacks = [earlystopping])
