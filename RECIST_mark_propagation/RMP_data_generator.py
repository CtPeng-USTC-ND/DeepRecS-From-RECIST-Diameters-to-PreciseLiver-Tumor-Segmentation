from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
from RECIST_mark_propagation.htmapGene import *
from skimage.measure import label, regionprops
import random

filelist = []

def trainGenerator(train_path,batch_size, inres=(256, 256), outres=(64, 64), classes=4, num_stack=2):
    i=-1
    train_input = np.zeros(shape=(batch_size, inres[0], inres[1], 3), dtype=np.float)
    gt_heatmap = np.zeros(shape=(batch_size, outres[0], outres[1], classes), dtype=np.float)
    for root, dirs, files in os.walk(os.path.join(train_path,'CT')):
        while True:
            for name in files:
                i=i+1
                _index=i%batch_size
                img = cv2.imread(os.path.join(root,name))
                img, gtmap = generate_gtmap(trans=True, img=img, path=train_path, name=name)
                img = img / 255.

                train_input[_index, :, :, :] = img
                gt_heatmap[_index, :, :, :] = gtmap

                if i%batch_size == (batch_size-1):
                    out_maps=[]
                    for m in range(num_stack):
                        out_maps.append(gt_heatmap)
                    yield (train_input, out_maps)

def devGenerator(dev_path, batch_size, inres=(256, 256), outres=(64, 64), classes=4, num_stack=2):
    i=-1
    train_input = np.zeros(shape=(batch_size, inres[0], inres[1], 3), dtype=np.float)
    gt_heatmap = np.zeros(shape=(batch_size, outres[0], outres[1], classes), dtype=np.float)
    for root, dirs, files in os.walk(os.path.join(dev_path, 'CT')):
        while True:
            for name in files:
                i=i+1
                _index=i%batch_size
                img = cv2.imread(os.path.join(root, name))
                img, gtmap = generate_gtmap(trans=False, img=img, path=dev_path, name=name)
                img = img / 255.

                train_input[_index, :, :, :] = img
                gt_heatmap[_index, :, :, :] = gtmap

                if i%batch_size == (batch_size-1):
                    out_maps=[]
                    for m in range(num_stack):
                        out_maps.append(gt_heatmap)
                    yield (train_input, out_maps)


def testGenerator(test_path):
    for root, dirs, files in os.walk(test_path):
        while True:
            for name in files:
                filelist.append(name)
                img = cv2.imread(os.path.join(test_path,name))
                img = img / 255.

                img = np.reshape(img, (1,) + img.shape)
                yield img

def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile[1]):
        pts=[]
        res = np.zeros((256, 256))
        for j in range(4):
            img = item[:,:,j]
            # max=np.max(img)
            # res1[img>=max]=255
            t = np.where(img==np.max(img))
            pts.append((t[1][0]*4, t[0][0]*4))

        cv2.line(res, pts[0], pts[1], color=(0,0,255), thickness=1)
        cv2.line(res, pts[2], pts[3], color=(0, 0, 255), thickness=1)
        cv2.imwrite("%s/%s.png"%(save_path,filelist[i][:-4]),res)
