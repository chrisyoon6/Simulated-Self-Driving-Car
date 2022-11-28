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

red_low = [0, 50, 50]
red_up = [10, 255, 255]
blue_low = [110, 50, 50]
blue_up = [130, 255, 255]
white_low = [0, 0, 100]
white_up = [179, 10, 255]

uh = 179
us = 10
uv = 255
lh = 0
ls = 0
lv = 100
lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    """Currently filtered images"""    
    self.blue_im = None
    self.red_im = None
    self.white_im = None

  def process_image(self,image):
    """Filters an image to show: blue, red, white only, respectively. Updates this object"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    self.blue_im = filter(hsv, blue_low, blue_up)
    self.red_im = filter(hsv, red_low, red_up)
    self.white_im = filter(hsv, white_low, white_up)

  def filter(self, image, hsv_low, hsv_up):
    """Filters the image to the hsv ranges specified"""
    mask = cv2.inRange(image, hsv_low, hsv_up)
    blur = cv2.GaussianBlur(mask,(3,3), 0)
    return blur
    
  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    processed_im = self.process_image(cv_image)
    print(processed_im)

    # draw contours on the original image
    # contours, hierarchy = cv2.findContours(image=processed_im, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # largest_item= sorted(contours, key=cv2.contourArea, reverse= True)[0]
    # M = cv2.moments(largest_item)

    # cx = int(M['m10']/M['m00'])
    # cy = int(M['m01']/M['m00'])

    # disp = cv2.circle(processed_im, (cx, cy), 2, (0,255,0), 2)
    # cv2.drawContours(image=disp, contours=contours, contourIdx=0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # see the results
    #self.i+=1
    #w_title = ("none {}".format(self.i))
    cv2.imshow('original_view', cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV))
    cv2.imshow('script_view', processed_im)
    cv2.waitKey(3)


def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)