import skimage.io as io
import os
import cv2
import numpy as np
from numpy import random
from skimage.measure import label, regionprops

def gaussian(img, pt, sigma):
    # Draw a 2D gaussian

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        print('gaussian error')
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def generate_gtmap(trans, img, path, name, sigma=3, inres = (256,256), outres=(64,64)):
    img_t = np.eye(3, dtype='float32')
    kpt_t = np.eye(3, dtype='float32')
    if trans:
        img_t, kpt_t = get_transform(inres, outres)
    T = img_t[:2, :]
    img = cv2.warpAffine(img,T,inres)
    gtmap = np.zeros(shape=(outres[0], outres[1], 4), dtype=float)
    txtname=name[:-4]+'.txt'
    i=-1
    pts=[]
    for line in open(os.path.join(path, 'kpt', txtname)):
        i=i+1

        if i ==3:
            [ptr, ptc] = line.split(',')
        else:
            [ptr,ptc]=line[:-1].split(',')
        pt = [int(int(ptc)/4), int(int(ptr)/4)]
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(kpt_t, new_pt)
        pts.append(new_pt[:2].astype(int))
    long_l, long_r, short_l, short_r = ([0,0],[0,0],[0,0],[0,0])

    if pts[0][1]<pts[1][1]:
        long_l = pts[0]
        long_r = pts[1]
    else:
        long_l = pts[1]
        long_r = pts[0]
    if pts[2][1]<pts[3][1]:
        short_l = pts[2]
        short_r = pts[3]
    else:
        short_l = pts[3]
        short_r = pts[2]

    ct0 = np.zeros((outres[0], outres[1]), dtype=float)
    gtmap[:, :, 0] = gaussian(ct0, long_l, sigma)
    ct1 = np.zeros((outres[0], outres[1]), dtype=float)
    gtmap[:, :, 1] = gaussian(ct1, long_r, sigma)
    ct2 = np.zeros((outres[0], outres[1]), dtype=float)
    gtmap[:, :, 2] = gaussian(ct2, short_l, sigma)
    ct3 = np.zeros((outres[0], outres[1]), dtype=float)
    gtmap[:, :, 3] = gaussian(ct3, short_r, sigma)

    return img, gtmap

def _get_transform(rot, scale, res):
    t = np.eye(3, dtype='float32')
    R = cv2.getRotationMatrix2D((res[0] / 2, res[0] / 2), rot, scale)
    tmp = np.eye(3, dtype='float32')
    tmp[:2,:] = R
    t = np.dot(tmp, t)
    #S=cv2.getRotationMatrix2D((res[0] / 2, res[0] / 2), 0, )

    return t

def get_transform(inres= (256, 256), outres=(64,64)):
    rot,scale = (0, 1.0)
    if random.choice([0, 1]):
        rot = np.random.randint(-1 * 30, 31)
    if random.choice([0, 1]):
        scale = np.random.uniform(0.8, 1.2)
    # print(flip,translatex, translatey, rot, scale)
    img_t = _get_transform(rot,scale, inres)
    kpt_t = _get_transform(rot,scale, outres)

    return img_t, kpt_t
