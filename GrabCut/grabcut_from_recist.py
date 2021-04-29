import numpy as np
import cv2
import os
from skimage import morphology
import matplotlib.pyplot as plt

def gc_from_recist(img_path, recist_path, save_path):
    recist_img = cv2.imread(recist_path, cv2.IMREAD_GRAYSCALE)
    img=cv2.imread(img_path)
    mask = np.zeros((256, 256), dtype="uint8")

    # min_x, min_y, max_x, max_y = (min(x_list), min(y_list), max(x_list), max(y_list))
    contours_, hierarchy = cv2.findContours(recist_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour=contours_[0]
    left_most = tuple(contour[contour[:, :, 0].argmin()][0])
    right_most = tuple(contour[contour[:, :, 0].argmax()][0])
    top_most = tuple(contour[contour[:, :, 1].argmin()][0])
    bottom_most = tuple(contour[contour[:, :, 1].argmax()][0])
    pts=[]
    pts.append(left_most)
    pts.append(right_most)
    pts.append(top_most)
    pts.append(bottom_most)
    x_list = []
    y_list = []
    for p in pts:
        x_list.append(p[0])
        y_list.append(p[1])
    min_x, min_y, max_x, max_y = (min(x_list), min(y_list), max(x_list), max(y_list))
    width_s = max_x - min_x
    height_s = max_y - min_y
    width_l = int(width_s * 1.3)
    height_l = int(height_s * 1.3)
    min_x = int(min_x - (width_l - width_s) / 2)
    max_x = int(min_x + width_l)
    min_y = int(min_y - (height_l - height_s) / 2)
    max_y = int(min_y + height_l)
    cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), color=(100, 100, 100), thickness=-1)
    # cv2.line(mask, pts[0], pts[1], thickness=3, color=(255, 255, 255))
    # cv2.line(mask, pts[2], pts[3], thickness=3, color=(255, 255, 255))
    mask[recist_img==255]=255

    img_dilated = np.zeros((256, 256), dtype="uint8")
    img_dilated[mask == 255] = 200
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize=(3, 3))
    img_dilated = cv2.dilate(img_dilated, kernel=kernel, iterations=18)
    poses = []
    for r in range(256):
        for c in range(256):
            if img_dilated[r][c] == 200:
                poses.append((r, c))
    for pos in poses:
        if mask[pos[0]][pos[1]] == 100:
            if mask[pos[0]][pos[1]] != 255:
                mask[pos[0]][pos[1]] = 200

    # cv2.imshow(' ',mask)
    # cv2.waitKey(0)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    mask[mask == 255] = cv2.GC_FGD
    mask[mask == 200] = cv2.GC_PR_FGD
    mask[mask == 100] = cv2.GC_PR_BGD
    mask[mask == 0] = cv2.GC_BGD

    cv2.grabCut(img, mask, None, bgdModel, fgdModel, 20, cv2.GC_INIT_WITH_MASK)

    tumor_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")
    # img = img * mask2[:, :, np.newaxis]

    cv2.imwrite(save_path, tumor_mask)
