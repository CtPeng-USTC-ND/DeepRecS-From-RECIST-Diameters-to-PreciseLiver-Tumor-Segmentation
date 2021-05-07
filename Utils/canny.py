import cv2
import os

read_dir = '/data1/djh/patch_1.5/test/label'
save_dir = '/data1/djh/patch_1.5/test/shape'
for file in os.listdir(read_dir):
    img = cv2.imread(os.path.join(read_dir, file), 0)
    canny = cv2.Canny(img, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    out = cv2.dilate(canny, kernel, iterations=1)
    blur = cv2.GaussianBlur(out, (3,3), 0)
    cv2.imwrite(os.path.join(save_dir, file), blur)
