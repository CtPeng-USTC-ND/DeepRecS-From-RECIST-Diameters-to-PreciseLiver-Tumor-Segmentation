import cv2
import os
import numpy as np
import csv
import shutil
import math
import numpy as np
from scipy.ndimage import morphology

def surfd(input1, input2, sampling=1, connectivity=1):
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1^morphology.binary_erosion(input_1, conn)
    Sprime = input_2^morphology.binary_erosion(input_2, conn)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])
    return sds

def rmsd_compute(pred,lb):
    surf_dis = surfd(pred, lb)
    out = 0.
    for item in surf_dis:
        out = out + item * item
    out = out / len(surf_dis)
    rmsd = math.sqrt(out)
    return rmsd

def msd_compute(pred,lb):
    surf_dis = surfd(pred, lb)
    msd = surf_dis.max()
    return msd

def asd_compute(pred, lb):
    surf_dis = surfd(pred, lb)
    asd = surf_dis.mean()
    return asd

def rvd_compute(pred, lb):
    rvd=(np.sum(pred)-np.sum(lb))/np.sum(lb)
    return rvd*100

def voe_compute(pred, lb):
    inter=(pred.astype(np.int)&lb.astype(np.int)).sum()
    voe=1-(inter/((pred.astype(np.int)|lb.astype(np.int)).sum()))
    return voe*100

def dice_compute(pred, lb):
    inter=(pred.astype(np.int)&lb.astype(np.int)).sum()
    dice=2*(inter/(np.sum(pred)+np.sum(lb)))
    return dice*100
