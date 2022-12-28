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
    plate_low = [0, 0, 90]
    plate_up = [179, 10, 210]

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
        self.plate_im = None

    def process_image(self, image):
        """Filters an image to show: blue, red, white only, respectively. Updates this object

        Args:
            image (cv::Mat): image to be processed 
        """        
        self.blue_im = ImageProcessor.filter(image, ImageProcessor.blue_low, ImageProcessor.blue_up)
        self.red_im = ImageProcessor.filter(image, ImageProcessor.red_low, ImageProcessor.red_up)
        self.white_im = ImageProcessor.filter(image, ImageProcessor.white_low, ImageProcessor.white_up)
        self.plate_im = ImageProcessor.filter_plate(image, ImageProcessor.plate_low, ImageProcessor.plate_up)

    @staticmethod
    def filter(image, hsv_low, hsv_up, type="bgr"):
        """Filters an RGB image within an hsv range, and applies a blur. Returns the resulting binary image

        Args:
            image (cv::Mat): RGB image data to filter
            hsv_low (list[int]): a list of the lower bound of the hue, saturation, value  
            hsv_up (list[int]): a list of the upper bound of the hue, saturation, value
            type (str): (Optional) the channel type of the image data. Assumed to be "bgr"
        Returns:
            cv::Mat: the procesed image (binary image)
        """        """"""
        if type == "rgb":
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if type == "bgr":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_up))
        blur = cv2.GaussianBlur(mask, (3, 3), 0)
        return blur
    
    @staticmethod 
    def filter_red(img, type="bgr"):
        return ImageProcessor.filter(img, ImageProcessor.red_low, ImageProcessor.red_up, type)
    @staticmethod
    def filter_blue(img, type="bgr"):
        return ImageProcessor.filter(img, ImageProcessor.blue_low, ImageProcessor.blue_up, type)
    @staticmethod
    def compare_frames(bin_img1, bin_img2):
        """Compares two binary images and calculates the difference between the images
        
        Args:
            bin_img1 (cv::Mat): binary image to compare
            bin_img2 (cv::Mat): binary image to compare

        Returns:
            float: the error between the images
        """        
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

    @staticmethod
    def filter_plate(image, hsv_low, hsv_up):
        """Filters and blurs the license plate image to the hsv ranges specified

        Args:
            image (cv::Mat): image respresenting the license plate
            hsv_low (list[int]): a list of the lower bound of the hue, saturation, value  
            hsv_up (list[int]): a list of the upper bound of the hue, saturation, value

        Returns:
            cv::Mat: the processed image containing the license plate.
        """        """"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_up))
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        dil = cv2.erode(blur, (9, 9))
        return dil

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
        self.truck_test(cv_image)

    def contours_area(img,nums=1):
        contours, hierarchy = cv2.findContours(
            image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        cs = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(cs) > nums:
            cs = cs[:nums]
        areas = [cv2.contourArea(c) for c in cs] 
        return areas
    
    def blue_area(self, cv_image):
        crped = ImageProcessor.crop(cv_image, row_start=int(720/2.2))
        blu_crped = ImageProcessor.filter_blue(crped)
        cv2.imshow("Blue", blu_crped)
        cv2.waitKey(1)
        blu_area = ImageProcessor.contours_area(blu_crped)[0]
        print(blu_area)

    def test_hugh_trans(self, img):
        bin = ImageProcessor.filter(img, ImageProcessor.red_low, ImageProcessor.red_up)
        cv2.imshow('script_view', bin)
        cv2.waitKey(1)
        edges = cv2.Canny(bin,50,150,apertureSize = 3)
        minLineLength=100
        lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)
        if list(lines):
            x1,y1,x2,y2 = lines[0][0].tolist()
            if x1 != x2:
                print(np.rad2deg(np.arctan((y2-y1)/(x2-x1))))
                print(y1,y2)
                print("")

    def truck_test(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = ImageProcessor.crop(img_gray, int(720/3), int(2*720/3), int(1280/2.65), int(2*1280/2.65))
        self.process_image(img) 
        mse = -1
        if self.temp_im is not None:
            mse = ImageProcessor.compare_frames(self.temp_im, img_gray)
        print("mse:", mse)
        self.temp_im = img_gray
        cv2.imshow('script_view', img_gray)
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
