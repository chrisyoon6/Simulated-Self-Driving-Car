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
from contour_approx import contour_approximator
from skimage.metrics import mean_squared_error

class ImageProcessor:
    """This class handles any image processing-related needs.
    """    

    red_low = [0, 50, 50]
    red_up = [10, 255, 255]
    blue_low = [110, 50, 50]
    blue_up = [130, 255, 255]
    white_low = [0, 0, 100]
    white_up = [179, 10, 255]

    def __init__(self):
        """Creates an ImageProcessor Object.
        """        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        """Currently filtered images"""
        self.blue_im = None
        self.red_im = None
        self.white_im = None
        self.temp_im = None

    def process_image(self, image):
        """Filters an image to show: blue, red, white only, respectively. Updates this object

        Args:
            image (cv::Mat): image to be processed 
        """        
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
    
    @staticmethod
    def compare_frames(bin_img1, bin_img2):
        return mean_squared_error(bin_img1, bin_img2)

    @staticmethod
    def crop(img, row_start=-1, row_end=-1, col_start=-1, col_end=-1):
        """Crops the image to row,columns within the specified range. 
        (Default) -1 if no change in the parameter.

        Args:
            img (cv::Mat): image to be cropped
            row_start (int): new starting row pixel
            row_end (int): new end row pixel (not inclusive)
            col_start (int): new starting column pixel
            col_end (int): new end column pixel (not inclusive)

        Returns:
            cv::Mat: cropped image
        """
        rows,cols,*rest = img.shape
        if row_start == -1:
            row_start = 0
        if row_end == -1:
            row_end = rows
        if col_start == -1:
            col_start = 0
        if col_end == -1:
            col_end = cols
        return img[row_start:row_end, col_start:col_end]


    def callback(self, data):
        """Callback function for the subscriber node for the /R1/.../image_raw ros topic. This callback is called 
        when a new message (frame) has arrived to the topic. 

        Processes the arrived image data into hsv form and updates this object.

        Args:
            data (sensor_msgs::Image): image data from the /R1/.../image_raw ros topic
        """        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        img_gray = ImageProcessor.crop(img_gray, 180, 720-180, 320, 1280-320)
        # print(cv_image.shape)
        self.process_image(cv_image) 
        # print(self.temp_im)
        mse = -1
        if self.temp_im is not None:
            mse = ImageProcessor.compare_frames(self.temp_im, img_gray)
        contours = contour_approximator.get_contours_area(self.red_im, 3)
        print("Contours:", contours)
        # print("mse:", mse)
        self.temp_im = img_gray
        # cv2.imshow('script_view', img_gray)
        cv2.imshow('script_view', self.red_im)
        
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
