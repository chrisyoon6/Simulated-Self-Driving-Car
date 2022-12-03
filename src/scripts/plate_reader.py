#! /usr/bin/env python3

from __future__ import print_function
from concurrent.futures import process

#import roslib; roslib.load_manifest('node')
import sys
import rospy
import cv2
import random
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from char_reader import CharReader
from hsv_view import ImageProcessor

# license plate working values

CAR_WIDTH = 200
CAR_HEIGHT = 320
PLATE_F = 270
PLATE_I = 220
PLATE_RES = (150, 298)
PATH_NUM_MODEL = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/models/num_model1.h5'
PATH_ALPHA_MODEL = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/models/alpha_model1.h5'
font = cv2.FONT_HERSHEY_COMPLEX
font_size = 0.5


class PlateReader:
    """This class handles license plate recognition.
    """

    def __init__(self):
        # self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback)
        self.num_reader = CharReader(PATH_NUM_MODEL)
        self.alpha_reader = CharReader(PATH_ALPHA_MODEL)
        self.i = 0

    def get_moments(self, img, debug=False):
        """Returns the moment of an image: c, cx, cy. 

        Usually cx, cy are only important for debugging
        c is the largest contour; 
        cx, cy is the center of mass of the largest contour

        Args:
            image (cv::Mat): image that is thresholded to be processed 
            debug (bool): if true, returns cx, cy as well
        Returns:
            list[float]: a list of the largest contours

        """
        
        contours, hierarchy = cv2.findContours(
            image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # gets the biggest contour and its info
        if not contours:
            return []
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        # print("Max area", M['m00'])

        if debug:
            return c, cx, cy

        return c

    def callback(self, data):
        """Handler of every callback when a new frame comes in

        Args:
            data (Image): For every new image, process that checks and reads plate
        """        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        out = cv_image.copy()
        processed_im = ImageProcessor.filter_plate(
            cv_image, ImageProcessor.plate_low, ImageProcessor.plate_up)

        c = self.get_moments(processed_im)

        if not list(c):
            return

        # cv2.imshow('contours', cv2.drawContours(cv2.resize(cv_image, (400, 300)), c, -1, (0,0,255), 3))
        cv2.waitKey(3)

        approx = self.approximate_plate(c, epsilon=0.1)

        verticies = self.verticies(approx_c=approx)

        # print("Verticies", verticies)

        if not list(verticies):
            print("No perspective transform")
            return

        plate_view = self.transform_perspective(
            CAR_WIDTH, CAR_HEIGHT, verticies, out)

        char_imgs = self.get_char_imgs(plate=plate_view)


        print(self.characters(char_imgs))

    def get_license_plate(self,img):
        processed_im = ImageProcessor.filter_plate(img, ImageProcessor.plate_low, ImageProcessor.plate_up)
        c = self.get_moments(processed_im)
        if not list(c):
            # no contour
            return ""
        approx = self.approximate_plate(c, epsilon=0.1)
        verticies = self.verticies(approx_c=approx)

        if not list(verticies):
            # no verticies (i.e. no perspec. transform)
            return ""
        plate_view = self.transform_perspective(CAR_WIDTH, CAR_HEIGHT, verticies, img)
        char_imgs = self.get_char_imgs(plate=plate_view)
        return self.characters(char_imgs)


    def characters(self, char_imgs):
        """Gets the neural network predicted characters from the images of each character.

        Args:
            char_imgs (array[Image]): Array (length 4) of character images from the license plate.
                First two images should be of letters, second two should be of numbers.

        Returns:
            str: a string representing the license plate
        """

        license_plate = ''
        for index,img in enumerate(char_imgs):
            if index < 2:
                prediction_vec = self.alpha_reader.predict_char(img=img)
                license_plate += CharReader.interpret(predict_vec=prediction_vec)
            else:
                prediction_vec = self.num_reader.predict_char(img=img)
                license_plate += CharReader.interpret(predict_vec=prediction_vec)

        return license_plate

    def get_char_imgs(self, plate):
        """Gets the verticies of a simple shape such as a square, rectangle, etc.

        Args:
            plate (Image): rectangular image of the license plate
        """
        imgs = []
        for i in range(4):
            imgs.append(self.process_plate(i, plate))
        return imgs

    def verticies(self, approx_c):
        """Gets the verticies of a simple contour such as a square, rectangle, etc.

        Args:
            approx_c (***): approximated contour of which the verticies are found
        """
        n = approx_c.ravel()
        pts = np.float32(self.get_coords(n)).reshape(-1, 2)
        sorted_pts = self.contour_coords_sorted(pts)
        return sorted_pts

    def approximate_plate(self, contour, epsilon):
        """Approximates a contour to a simple shape such as a square, rectangle, etc.

        Args:
            contour (***): contour to be approximated
            epsilon (float in (0,1)): approximation accuracy
        """
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon*perimeter, True)
        return approx

    def process_plate(self, pos, plate_im):
        """Crops and processes plate images for individual letter.

        Args: pos - the position in the license plate
              plate_im - image of license plate

        Returns: processed image of plate"""

        crop = plate_im[PLATE_I:PLATE_F, int(
            pos*CAR_WIDTH/4):int((pos + 1)*CAR_WIDTH/4)]
        resize = cv2.resize(crop, PLATE_RES)

        return resize

    def transform_perspective(self, width, height, sorted_pts, image):
        """Args: The coords of the polygon we are to transform into a rectangle.
                 Desired width and height of the transformed image.
                 The image from which we pull the polygon.

                 Returns: The polygon from the original image transformed into a square."""
        pts = np.float32([[0, 0], [width, 0],
                          [0, height], [width, height]])
        # print("transform perspective")
        Mat = cv2.getPerspectiveTransform(sorted_pts, pts)
        return cv2.warpPerspective(image, Mat, (width, height))

    def get_coords(self, contour):
        """Args: Approximated contour extracted with CHAIN_APPROX_NONE (only the verticies)
           Returns: List of verticies in (x,y) coords"""
        i = 0
        coords = []
        for j in contour:
            if (i % 2 == 0):
                x = contour[i]
                y = contour[i + 1]
                coords.append((x, y))

            i = i + 1

        return coords

    def contour_coords_sorted(self, list_of_points):
        """Sorts the verticies of a contour so it can be perspective transformed
        
        Args: List of contour verticies. Should have exactly 4 verticies with (x,y)
        
        Returns: Verticies in list sorted by top to bottom, left to right"""

        avg_y = 0
        avg_x = 0
        # print("List of points:", list_of_points)
        for i in list_of_points:
            avg_y += i[1]
            avg_x += i[0]

        avg_y = int(avg_y/4)
        avg_x = int(avg_x/4)

        tl = tr = bl = br = None

        for i in list_of_points:
            if (int(i[1]) < avg_y and int(i[0]) < avg_x):
                tl = i
            elif (int(i[1]) < avg_y):
                tr = i
            elif (int(i[0]) < avg_x):
                bl = i
            else:
                br = i
                
        if tl is None or tr is None or bl is None or br is None:
            return []

        tl = list(tl)
        tr = list(tr)
        bl = list(bl)
        br = list(br)

        coords = [tl, tr, bl, br]
        return np.float32(coords).reshape(-1, 2)


def main(args):
    pr = PlateReader()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
