import cv2
import numpy as np

def FillHole(imgPath, SavePath):
    im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    im_floodfill = im_in.copy()

    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if isbreak:
            break

    cv2.floodFill(im_floodfill, mask, seedPoint, 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_in | im_floodfill_inv
    cv2.imwrite(SavePath, im_out)