# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas
from PIL import Image
import cv2
import seaborn


class CAM:
    def __init__(self, model_path, base_model='VGG16', shape=(224,224,3)):
        self.model = tf.keras.models.load_model(model_path)
        self.base_model = tf.keras.applications.VGG16(input_shape=shape, include_top=False, weights='imagenet')
        self.heatmap = None
        
    def __get_heatmap(self, img, size=(224,224)):
        '''
        img: input original image, tensor, 4 dims
        size: input image size, size for heatmap to be generated
        '''        
        feat_map = self.base_model(img)
        pred = self.model.predict(img)
        index = tf.argmax(pred)
        weights = self.model.get_layer('dense').get_weights()[0][:,index]
        heatmap = tf.matmul(feat_map, weights)
        heatmap = np.array(heatmap[0])
        heatmap = (heatmap-heatmap.min()) / (heatmap.max()-heatmap.min())
        heatmap = cv2.resize(heatmap, size)
        heatmap = heatmap * 255
        heatmap = np.array(heatmap, dtype=np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        self.heatmap = heatmap
        return heatmap
        
    def plot_CAM(self, img, size=(224,224), resize=(224,224)):
        '''
        img: input original image, tensor, 4 dims
        size: input image size
        resize: the size for output CAM image
        '''        
        if self.heatmap is None:
            self.__get_heatmap(img, size)
        img = np.array(img[0], dtype=np.uint8)
        cam = cv2.addWeighted(src1=img, alpha=0.7, src2=self.heatmap, beta=0.4, gamma=0)
        cam = cv2.resize(cam, dsize=size)
        plt.imshow(cam[:,:,::-1])
        plt.show()
        
    def save_CAM(self, img, dest_path, size=(224,224), resize=(224,224)):
        '''
        img: input original image, tensor, 4 dims
        size: input image size
        resize: the size for output CAM image
        '''        
        if self.heatmap is None:
            self.__get_heatmap(img, size)
        img = np.array(img[0], dtype=np.uint8)
        cam = cv2.addWeighted(src1=img, alpha=0.7, src2=self.heatmap, beta=0.4, gamma=0)
        cam = cv2.resize(cam, dsize=size)
        cv2.imwrite(dest_path, cam)
        
