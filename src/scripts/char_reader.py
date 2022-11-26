#! /usr/bin/env python3

from __future__ import print_function
from concurrent.futures import process

#import roslib; roslib.load_manifest('node')
import sys
import rospy
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.utils import plot_model
from PIL import Image


class char_reader:
    def __init__(self, path):
        self.model = models.load_model(path)
        print(type(self.model))

    def predict(self, img, debug=False):
        """Returns the model predicted character for a given image.
        If `debug=True` it returns the probability too."""

        img_aug = np.expand_dims(img, axis=0)
        y_predict = self.model.predict(img_aug)[0]

        if np.argmax(y_predict) < 26:
            char = chr(np.argmax(y_predict)+ord("A"))
        else:
            char = chr(np.argmax(y_predict)-26+ord("0"))

        if debug:
            prob = y_predict[np.argmax(y_predict)]
            return char, prob

        return char

    def model_summary(self):
        """Returns model summary"""
        return self.model.summary()


def main(args):
    path = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/models/license_plate_model1.h5'
    cr = char_reader(path)

    input = np.array(Image.open('/home/fizzer/ros_ws/src/ENPH353-Team12/src/license-plate-data/test_char_Y.png'))
    input = input/255
    input = np.expand_dims(input, axis=0)

    print(input.shape)
    print(cr.predict(input))

    rospy.init_node('char_reader', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/models/license_plate_model1.h5'
    model = models.load_model(path)

    # main(sys.argv)
