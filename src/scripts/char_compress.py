#! /usr/bin/env python3

import cv2
import numpy as np
import os
from PIL import Image

directory = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/alpha-edge-data/'
directory_id = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/id-data/'
save_path_id = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/id-data-compressed/'
save_path_num = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/num-edge-data-compressed/'
save_path_alpha = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/alpha-edge-data-compressed/'

# count = 0

for filename in os.listdir(directory_id): # CHANGE THIS
    f = os.path.join(directory, filename)
    print(filename)
    # checking if it is a file
    if os.path.isfile(f):
        # count += 1
        im = np.array(Image.open(f))
        cv2.imshow('im', im)
        
        scale_percent = 10 # percent of original size
        width = int(im.shape[1] * scale_percent / 100)
        height = int(im.shape[0] * scale_percent / 100)
        
        dim = (width, height)
        resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
        
        # cv2.imwrite(os.path.join(save_path_id, filename), resized)

        # if (filename[6].isalpha()):
        #     cv2.imwrite(os.path.join(save_path_alpha, filename), resized)
        # else:
        #     cv2.imwrite(os.path.join(save_path_num, filename), resized)

        # cv2.imshow('image', resized)
        cv2.waitKey(3)

# print('File count: ' + str(count))
