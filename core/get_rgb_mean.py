import random
import sys
import glob
import os
import numpy as np
import cv2
# from core.mask_rcnn_config import TRAINING_DATA_DIR


def get_mean(nr_images):
    path = r"D:\training_images\_mapping-challenge\val\images"
    # path = TRAINING_DATA_DIR
    images = glob.glob(os.path.join(path, "**/*.jpg"), recursive=True)
    random.shuffle(images)

    # all_rgb = np.zeros((1,1,1,len(images[:nr_images])))
    r_tot = 0
    g_tot = 0
    b_tot = 0
    avg = np.zeros((1,1,1,len(images[:nr_images])))
    for i,img_path in enumerate(images[:nr_images]):
        im = cv2.imread(img_path)
        r, g, b = np.array(im).mean(axis=(0, 1))
        r_tot += r
        g_tot += g
        b_tot += b
    r_tot /= len(images[:nr_images])
    g_tot /= len(images[:nr_images])
    b_tot /= len(images[:nr_images])
    print(r_tot, g_tot, b_tot)

    # im = Image.open('image.gif')
    # rgb_im = im.convert('RGB')
    # r, g, b = rgb_im.getpixel((1, 1))
    # pass


if __name__ == "__main__":
    nr_images = 10000
    if len(sys.argv) > 1:
        nr_images = int(sys.argv[1])
    get_mean(nr_images)
