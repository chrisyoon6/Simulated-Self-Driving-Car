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

uh = 179
us = 10
uv = 230
lh = 0
ls = 0
lv = 150
lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    self.i = 0


  def process_image(self,image):
    #image processing
    # image[:,:,2] = 0
    # image[:,:,1] = 0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    blur = cv2.GaussianBlur(mask,(7,7), 0)

    return blur

    
  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    out = cv_image.copy()
    processed_im = self.process_image(cv_image)

    # draw contours on the original image

    # gets the contours in the thresholded image
    contours, hierarchy = cv2.findContours(image=processed_im, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    #gets the biggest contour and its info
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)

    # gets the center of mass of the contour
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # draws a circle at the center of mass of contour
    disp = cv2.circle(out, (cx, cy), 2, (0,255,0), 2)

    # approximates the contour to a simpler shape
    epsilon = 0.1 #higher means simplify more
    perimiter = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,epsilon*perimiter,True)


    cv2.drawContours(image=disp, contours=[approx], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
 
    cv2.imshow('script_view', disp)
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