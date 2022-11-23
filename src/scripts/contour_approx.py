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
uv = 210
lh = 0
ls = 0
lv = 90
lower_hsv = np.array([lh,ls,lv])
upper_hsv = np.array([uh,us,uv])

CAR_WIDTH = 200
CAR_HEIGHT = 320
PLATE_F = 270
PLATE_I = 220
PLATE_RES = (150, 298)

font = cv2.FONT_HERSHEY_COMPLEX
font_size = 0.5

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
    self.i = 0


  def process_stream(self,image):
    """processes the image using a grey filter to catch license plates
    
    returns a cv image"""
    #image processing
    # image[:,:,2] = 0
    # image[:,:,1] = 0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    blur = cv2.GaussianBlur(mask,(5,5), 0)
    dil = cv2.dilate(blur, (5,5))

    return dil

    
  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    out = cv_image.copy()
    processed_im = self.process_stream(cv_image)

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

    n = approx.ravel()
    pts = np.float32(self.get_coords(n)).reshape(-1, 2)
    sorted_pts = self.contour_coords_sorted(pts)
    
    cv2.putText(disp, "tl", (int(sorted_pts[0][0]), int(sorted_pts[0][1])), font, font_size, (0, 255, 0)) 
    cv2.putText(disp, "tr", (int(sorted_pts[1][0]), int(sorted_pts[1][1])), font, font_size, (0, 255, 0)) 
    cv2.putText(disp, "bl", (int(sorted_pts[2][0]), int(sorted_pts[2][1])), font, font_size, (0, 255, 0)) 
    cv2.putText(disp, "br", (int(sorted_pts[3][0]), int(sorted_pts[3][1])), font, font_size, (0, 255, 0)) 
    # print(pts)

    # resizing to have pairs of points
    plate_view = self.transform_perspective(CAR_WIDTH, CAR_HEIGHT, sorted_pts, out)

    cv2.drawContours(image=disp, contours=[approx], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
 
    cv2.imshow('plate_view', plate_view)

    cv2.imshow('char 1', self.process_plate(0, plate_view))
    cv2.imshow('char 2', self.process_plate(1, plate_view))
    cv2.imshow('char 3', self.process_plate(2, plate_view))
    cv2.imshow('char 4', self.process_plate(3, plate_view))

    cv2.imshow('script_view', processed_im)
    cv2.waitKey(3)

  
  def process_plate(self, pos, plate_im):
    """Crops and processes plate images for individual letter.
    
    Args: pos - the position in the license plate
          plate_im - image of license plate
          
    Returns: processed image of plate"""

    crop = plate_im[PLATE_I:PLATE_F, int(pos*CAR_WIDTH/4):int((pos + 1)*CAR_WIDTH/4)]
    resize = cv2.resize(crop, PLATE_RES)

    return resize


  
  def transform_perspective(self, width, height, sorted_pts, image):
    """Args: The coords of the polygon we are to transform into a rectangle.
             Desired width and height of the transformed image.
             The image from which we pull the polygon.

             Returns: The polygon from the original image transformed into a square."""
    pts = np.float32([[0, 0], [width, 0],
                      [0, height], [width, height]])
    Mat = cv2.getPerspectiveTransform(sorted_pts, pts)
    return cv2.warpPerspective(image, Mat, (width, height))

  
  def get_coords(self, contour):
    """Args: Approximated contour extracted with CHAIN_APPROX_NONE (only the verticies)
       Returns: List of verticies in (x,y) coords"""
    i = 0
    coords = []
    for j in contour :
        if(i % 2 == 0):
            x = contour[i]
            y = contour[i + 1]
            coords.append((x,y))
  
        i = i + 1
      
    return coords


  def contour_coords_sorted(self, list_of_points):
    """Args: List of contour verticies
       Returns: Verticies in list sorted by top to bottom, left to right"""

    avg_y = 0
    avg_x = 0

    for i in list_of_points:
      avg_y += i[1]
      avg_x += i[0]

    avg_y = int(avg_y/4)
    avg_x = int(avg_x/4)
    

    for i in list_of_points:
      if (int(i[1]) < avg_y and int(i[0]) < avg_x):
        tl = i
      elif (int(i[1]) < avg_y):
        tr = i
      elif (int(i[0]) < avg_x):
        bl = i
      else:
        br = i

    coords = [list(tl), list(tr), list(bl), list(br)]

    return np.float32(coords).reshape(-1, 2)

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