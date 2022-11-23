#! /usr/bin/env python3

from __future__ import print_function
from concurrent.futures import process

#import roslib; roslib.load_manifest('node')
import sys
import rospy
import cv2
import time
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageProcessor:
    red_low = [0, 50, 50]
    red_up = [10, 255, 255]
    blue_low = [110, 50, 50]
    blue_up = [130, 255, 255]
    white_low = [0, 0, 200]
    white_up = [179, 10, 255]

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        """Currently filtered images"""
        self.blue_im = None
        self.red_im = None
        self.white_im = None

    def process_image(self, image):
        """Filters an image to show: blue, red, white only, respectively. Updates this object"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.blue_im = ImageProcessor.filter(hsv, ImageProcessor.blue_low, ImageProcessor.blue_up)
        self.red_im = ImageProcessor.filter(hsv, ImageProcessor.red_low, ImageProcessor.red_up)
        self.white_im = ImageProcessor.filter(hsv, ImageProcessor.white_low, ImageProcessor.white_up)

    @staticmethod
    def filter(image, hsv_low, hsv_up):
        """Filters the image to the hsv ranges specified"""
        mask = cv2.inRange(image, np.array(hsv_low), np.array(hsv_up))
        blur = cv2.GaussianBlur(mask, (3, 3), 0)
        return blur

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.process_image(cv_image)

        cv2.imshow('script_view', self.blue_im)
        cv2.waitKey(3)


def main(args):
    ic = ImageProcessor()
    rospy.init_node('ImageProcessor', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
