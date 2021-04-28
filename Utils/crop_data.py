import cv2
import numpy as np
import os
from skimage.measure import label, regionprops
import shutil
import math

def compute_key_slice(path):
    max_slice = 0
    max_size = 0
    for name in os.listdir(path):
        slice = name.split('_')[2][:-4]
        img = cv2.imread(os.path.join(path, name))
        if np.max(img) != np.min(img):
            size = np.sum(img == 255)
            if size > max_size:
                max_size = size
                max_slice = slice
    return max_slice,max_size

def unique_tumor_label(lb_path):
    lb = cv2.imread(lb_path, cv2.IMREAD_GRAYSCALE)
    img_lb = label(lb, 4)
    props = regionprops(img_lb)
    region = props[0]
    for item in props:
        if region.area < item.area:
            region = item

    lb_cur = np.zeros((512, 512))
    for coord in region.coords:
        lb_cur[coord[0], coord[1]] = 255
    return lb_cur

def compute_box(lb_path, f=1.0):
    lb = cv2.imread(lb_path, cv2.IMREAD_GRAYSCALE)

    img_lb = label(lb, 4)
    props = regionprops(img_lb)
    region = props[0]
    for item in props:
        if region.area<item.area:
            region = item

    lb_cur = np.zeros((512, 512))

    contour = []
    for coord in region.coords:
        lb_cur[coord[0], coord[1]] = 255
        if lb[coord[0], coord[1] + 1] == 0 or lb[coord[0], coord[1] - 1] == 0 or lb[
            coord[0] + 1, coord[1]] == 0 or lb[coord[0] - 1, coord[1]] == 0:
            contour.append((coord[1], coord[0]))
    if len(contour):
        [minx, miny, maxx, maxy] = [511, 511, 0, 0]

        for c in contour:
            if minx > c[0]:
                minx = c[0]
            if miny > c[1]:
                miny = c[1]
            if maxx < c[0]:
                maxx = c[0]
            if maxy < c[1]:
                maxy = c[1]
        minx = minx - 5  # margin
        miny = miny - 5
        maxx = maxx + 5
        maxy = maxy + 5
        h = maxy - miny
        w = maxx - minx
        if h > w:
            minx = int(minx - (h - w) / 2)
            maxx = minx + h
        if h < w:
            miny = int(miny - (w - h) / 2)
            maxy = miny + w
        # [x1, y1, x2, y2] = [minx, miny, maxx, maxy]
        l1 = maxy - miny

        minx = int(minx - l1 * f)
        maxx = int(maxx + l1 * f)
        miny = int(miny - l1 * f)
        maxy = int(miny + maxx - minx)
        l_nx = maxy - miny
        # cv2.rectangle(ct, (minx, miny), (maxx, maxy), (0, 0, 255), thickness=1)
        p = 0
        if minx < 0 or miny < 0 or maxx > 511 or maxy > 511:
            p = 0
            if 0 - minx > p:
                p = -minx  # p=abs(minx)
            if 0 - miny > p:  # abs(miny) > abs(minx)
                p = - miny
            if maxx - 511 > p:
                p = maxx - 511
            if maxy - 511 > p:
                p = maxy - 511
            # ct = cv2.copyMakeBorder(ct, p, p, p, p, cv2.BORDER_CONSTANT)
            # lb_cur = cv2.copyMakeBorder(lb_cur, p, p, p, p, cv2.BORDER_CONSTANT)
            miny = miny + p
            maxy = miny + l_nx
            minx = minx + p
            maxx = minx + l_nx
            return miny, maxy, minx, maxx, p

        else:
            return miny, maxy, minx, maxx, p

def crop(img_save_path, lb_save_path):
    dir = 'E:/2017_png/pos/label'
    for case in os.listdir(dir):

        path = os.path.join(dir, case)
        key_slice, max_size = compute_key_slice(path)
        file_name = case+'_'+str(key_slice)+'.png'
        key_slice_path = os.path.join(path, file_name)
        minr, maxr, minc, maxc, p= compute_box(key_slice_path, f=0.25)
        #f=0.25  1.5x; f=1.0 3x; f=0.5 2x;  f=1.75 4.5x
        # if p!=0:
        #     print(case+':'+str(p))
        for file in os.listdir(path):
            ct = cv2.imread(os.path.join(path.replace('label','CT'), file))
            lb = cv2.imread(os.path.join(path, file))
            # lb = unique_tumor_label(os.path.join(path, file))
            ct = cv2.copyMakeBorder(ct, p, p, p, p, cv2.BORDER_CONSTANT)
            lb = cv2.copyMakeBorder(lb, p, p, p, p, cv2.BORDER_CONSTANT)
            tumor = ct[minr:maxr, minc:maxc]
            lb = lb[minr:maxr, minc:maxc]

            length = 256
            tumor = cv2.resize(tumor, (length, length))
            lb = cv2.resize(lb, (length, length))

            lb[lb >= 210] = 255
            lb[lb < 210] = 0

            cv2.imwrite(os.path.join(lb_save_path, file), lb)
            cv2.imwrite(os.path.join(img_save_path, file), tumor)


