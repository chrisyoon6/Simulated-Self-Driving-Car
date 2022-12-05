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


class CharReader:
    """This class handles character prediction from neural net.

    Requires path of the neural net file
    """

    def __init__(self, path):
        self.model = models.load_model(path)
        print(type(self.model))

    def predict_char(self, img, id=False):
        """Model prediction vector for a given image.

        Returns:
            List: the prediction vector for each possible prediction outcome
        """
        if (id):
            img = self.pre_processing_for_id(img)
        else:
            img = self.pre_processing_for_model(img)
        img = img/255
        img_aug = np.expand_dims(np.expand_dims(img, axis=-1), axis=0)
        y_predict = self.model.predict(img_aug)[0]

        return y_predict

    @staticmethod
    def interpret(predict_vec, debug=False):
        """Converts prediction vector into character output

        Args:
            predict_vec (list): prediction vector given from neural net
            debug (bool, optional): if true, also returns the probabilities. Defaults to False.

        Returns:
            char: output character
            prob (optional): the probability of the top character prediction
        """
        print(sorted(predict_vec, reverse=True)[:2])
        if len(predict_vec) == 26:
            out = chr(np.argmax(predict_vec)+ord('A'))
        elif len(predict_vec) == 10:
            out = chr(np.argmax(predict_vec)+ord('0'))
        elif len(predict_vec) == 8:
            out = chr(np.argmax(predict_vec)+ord('1'))
        else:
            print('Invalid prediction vector')
            return

        if debug:
            prob = predict_vec[np.argmax(predict_vec)]
            return out, prob

        return out

    def pre_processing_for_id(self, im):
        """EXACT same as pre_processing_for_model
        but with size (15,30)

        Args:
            im (image): input image with a character to be read

        Returns:
            image: formatted image
        """

        resize = cv2.resize(im, (15, 30))
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

        return gray

    def pre_processing_for_model(self, im):
        """Formats image to dimensions for use in neural net

        Args:
            im (image): input image with a character to be read

        Returns:
            image: formatted image
        """
        resize = cv2.resize(im, (15, 29))
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

        return gray

    def model_summary(self):
        """Returns model summary"""
        return self.model.summary()


def main(args):
    path = '/home/fizzer/ros_ws/ENPH353-Team12/src/models/license_plate_model1.h5'
    print('***** initializing reader *****')
    cr = CharReader(path)

    input = np.array(Image.open(
        '/home/fizzer/ros_ws/src/ENPH353-Team12/src/license-plate-data/test_char_E.png'))
    print('***** input shape *****')
    print(input.shape)
    print('***** prediction output *****')
    print(cr.predict(input))

    # rospy.init_node('char_reader', anonymous=True)
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print("Shutting down")
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    path = '/home/fizzer/ros_ws/src/models/license_plate_model1.h5'
    print('***** loading model *****')
    model = models.load_model(path)

    main(sys.argv)
