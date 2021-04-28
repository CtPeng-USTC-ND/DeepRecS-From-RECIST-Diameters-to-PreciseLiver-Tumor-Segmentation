from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import skimage.io as io
import cv2

filelist = []

def adjustData(img1, img2, recist1, recist2, label, edge):
    input1 = np.concatenate((img1, recist1), axis=3)
    input2 = np.concatenate((img2, recist2), axis=3)
    input1 = input1 / 255
    input2 = input2 /255
    label = label /255
    edge = edge/255

    label[label > 0.5] = 1
    label[label <= 0.5] = 0
    return ( [input1, input2] , [label, edge])


def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        # img[img<0.5]=0
        # img[img>=0.5]=1

        io.imsave(os.path.join(save_path,"%s"%filelist[i]),img)
        # io.imsave(os.path.join(save_path, "%d.png"%(i) ), img)

def saveResult_2out(save_path, npyfile):
    for j,outs in enumerate(npyfile):
        if j==0:
            for i, item in enumerate(outs):
                img = item[:, :, 0]
                # img[img < 0.5] = 0
                # img[img >= 0.5] = 1

                io.imsave(os.path.join(save_path, "%s" % filelist[i]), img)

def trainGenerator(batch_size, data_path, folder, aug_dict, seed = 1, interaction='recist'):
    local_patch_gene = ImageDataGenerator(**aug_dict).flow_from_directory(
        data_path, classes=[folder], batch_size = batch_size,
        save_prefix=None, class_mode=None, seed = seed)
    global_patch_gene = ImageDataGenerator(**aug_dict).flow_from_directory(
        data_path.replace('CT_15', 'CT_45'), classes=[folder],
        batch_size = batch_size, class_mode=None, seed = seed)
    local_recist_gene = ImageDataGenerator(**aug_dict).flow_from_directory(
        data_path.replace('CT_15', interaction+'_15'), classes=[folder], color_mode='grayscale',
        batch_size=batch_size, class_mode=None, seed=seed)
    global_recist_gene = ImageDataGenerator(**aug_dict).flow_from_directory(
        data_path.replace('CT_15', interaction+'_45'), classes=[folder], color_mode='grayscale',
        batch_size=batch_size, class_mode=None, seed=seed)
    label_gene = ImageDataGenerator(**aug_dict).flow_from_directory(
        data_path.replace('CT_15', 'label_15'), classes=[folder], color_mode ='grayscale',
        batch_size = batch_size, class_mode=None, seed = seed)
    edge_gene = ImageDataGenerator(**aug_dict).flow_from_directory(
        data_path.replace('CT_15', 'edge_15'), classes=[folder], color_mode='grayscale',
        batch_size=batch_size, class_mode=None, seed=seed)

    train_generator = zip(local_patch_gene, global_patch_gene, local_recist_gene,
                          global_recist_gene, label_gene, edge_gene)
    while True:
        for (img1, img2, recist1, recist2, label, edge) in train_generator:
            input, gt = adjustData(img1, img2, recist1, recist2, label, edge)
            yield (input, gt)

def testGenerator(test_path, interaction='recist'):
    file_name = os.listdir(test_path)
    while True:
        for i in range(len(file_name)):
            filelist.append(file_name[i])
            img_local = cv2.imread(os.path.join(test_path, file_name[i]))
            img_global = cv2.imread(os.path.join(test_path.replace('CT_15', 'CT_45'), file_name[i]))
            recist_local = cv2.imread(os.path.join(test_path.replace('CT_15', interaction+'_15'),
                                                   file_name[i]), cv2.IMREAD_GRAYSCALE)
            recist_global = cv2.imread(os.path.join(test_path.replace('CT_15', interaction+'_45'),
                                                   file_name[i]), cv2.IMREAD_GRAYSCALE)
            # print(img_local.shape)
            # print(recist_local.shape)
            input1 = np.concatenate((img_local, recist_local[:, :, np.newaxis]), axis=2)
            input2 = np.concatenate((img_global, recist_global[:, :, np.newaxis]), axis=2)
            input1 = input1 / 255.
            input2 = input2 / 255.

            input1 = input1[np.newaxis, :, :, :]
            input2 = input2[np.newaxis, :, :, :]
            yield ([input1, input2])
