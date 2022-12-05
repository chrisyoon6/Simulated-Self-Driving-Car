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
from plate_reader import PlateReader

# license plate working values
uh = 179
us = 10
uv = 210
lh = 0
ls = 0
lv = 90
lower_hsv = np.array([lh, ls, lv])
upper_hsv = np.array([uh, us, uv])

CAR_WIDTH = 200
CAR_HEIGHT = 320
PLATE_F = 270
PLATE_I = 220
PLATE_RES = (150, 298)

ID_TOP = 130
ID_BOT = 185
ID_LEFT = 110
ID_RIGHT = 190

PATH_PARKING_ID = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/models/id_model1.h5'


font = cv2.FONT_HERSHEY_COMPLEX
font_size = 0.5


"""
Data collection pipeline:
for each character 0-9,A-Z:
    1) Light, dark lighting (differ locations - i.e. dark when under tree)
    2) Clear, blurry (distance from license plate, grass part more blurry)
    3) differ positions of the char (left 2 if letter, right 2 if num)

Around 36 * 2 * 2 * 2 = 288 frames 
"""

class PlatePull:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.callback)
        self.id_reader = CharReader(PATH_PARKING_ID)
        self.i = 0

    def process_stream(self, image):
        """processes the image using a grey filter to catch license plates
        returns a cv image"""

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        dil = cv2.dilate(blur, (5, 5))

        return dil

    @staticmethod
    def get_contours_area(img, nums=1):
        """Obtains the top contour areas that are present in the binary image.
        Defaulted to obtain the max.

        Args:
            img (cv::Mat): Binary image (i.e. two values)
            nums (int): number of top contour areas to obtain

        Returns:
            list(float): the top contour areas, sorted in descending order
        """        
        contours, hierarchy = cv2.findContours(
            image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        cs = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(cs) > nums:
            cs = cs[:nums]
        areas = [cv2.contourArea(c) for c in cs] 
        return areas

    def get_moments(self, img):
        """Returns c, cx, cy. (Usually cx, cy are only important for debugging text)
        c is the largest contour; 
        cx, cy is the center of mass of the largest contour"""
        contours, hierarchy = cv2.findContours(
            image=img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # gets the biggest contour and its info
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        return  c, cx, cy

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        out = cv_image.copy()
        processed_im = self.process_stream(cv_image)

        # draw contours on the original image

        c, cx, cy = self.get_moments(processed_im)

        # draws a circle at the center of mass of contour
        disp = cv2.circle(out, (cx, cy), 2, (0, 255, 0), 2)

        # approximates the contour to a simpler shape
        epsilon = 0.1  # higher means simplify more
        perimiter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon*perimiter, True)

        n = approx.ravel()
        pts = np.float32(self.get_coords(n)).reshape(-1, 2)
        # sorted_pts = self.contour_coords_sorted(pts)
        sorted_pts = PlateReader.contour_coords_sorted(pts)
        if not list(sorted_pts):
            return
        cv2.putText(disp, "tl", (int(sorted_pts[0][0]), int(
            sorted_pts[0][1])), font, font_size, (0, 255, 0))
        cv2.putText(disp, "tr", (int(sorted_pts[1][0]), int(
            sorted_pts[1][1])), font, font_size, (0, 255, 0))
        cv2.putText(disp, "bl", (int(sorted_pts[2][0]), int(
            sorted_pts[2][1])), font, font_size, (0, 255, 0))
        cv2.putText(disp, "br", (int(sorted_pts[3][0]), int(
            sorted_pts[3][1])), font, font_size, (0, 255, 0))
        # print(pts)

        # resizing to have pairs of points
        plate_view = self.transform_perspective(
            CAR_WIDTH, CAR_HEIGHT, sorted_pts, out)

        cv2.drawContours(image=disp, contours=[
                         approx], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # cv2.imshow('plate_view', plate_view)

        char_imgs = []
        for i in range(4):
          char_imgs.append(self.process_plate(i, plate_view))

        alpha_edge_PATH = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/alpha-edge-data/plate_'
        num_edge_PATH = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/num-edge-data/plate_'
        id_PATH = '/home/fizzer/ros_ws/src/id-data/carID_'

        # cv2.imshow('char 1', char_imgs[0])
        # cv2.imshow('char 2', char_imgs[1])
        # cv2.imshow('char 3', char_imgs[2])
        # cv2.imshow('char 4', char_imgs[3])
        r = random.random()
        # cv2.imwrite(alpha_edge_PATH + 'B' + str(r) + '.png', cv2.cvtColor(char_imgs[0], cv2.COLOR_BGR2GRAY))
        # cv2.imwrite(alpha_edge_PATH + 'O' + str(r) + '.png', cv2.cvtColor(char_imgs[1], cv2.COLOR_BGR2GRAY))        
        # cv2.imwrite(num_edge_PATH + '3' + str(r) + '.png', cv2.cvtColor(char_imgs[2], cv2.COLOR_BGR2GRAY))
        # cv2.imwrite(num_edge_PATH + '8' + str(r) + '.png', cv2.cvtColor(char_imgs[3], cv2.COLOR_BGR2GRAY))

        # cv2.imshow('plate_view', plate_view)
        plate_id = self.plate_id_img(plate_im=plate_view)
        print("\n")
        print("\n")
        print("\n")
        prediction_vec = self.id_reader.predict_char(img=plate_id, id=True)
        id = ''
        id += CharReader.interpret(predict_vec=prediction_vec)
        print(id)
        print(prediction_vec)

        cv2.imwrite(id_PATH + '1' + str(r) + '.png', cv2.cvtColor(plate_id, cv2.COLOR_BGR2GRAY))

        cv2.imshow('parking_id', plate_id)
        cv2.waitKey(3)

    def process_plate(self, pos, plate_im):
        """Crops and processes plate images for individual letter.
        Args: pos - the position in the license plate
              plate_im - image of license plate
        Returns: processed image of plate"""

        crop = plate_im[PLATE_I:PLATE_F, int(
            pos*CAR_WIDTH/4):int((pos + 1)*CAR_WIDTH/4)]
        resize = cv2.resize(crop, PLATE_RES)

        return resize

    def plate_id_img(self, plate_im):
        """Crops and processes plate images for parking ID.
        Args:
            plate_im (Image): image of the license plate
        Returns:
            Image: processed image of the parking ID
        """        
        crop = plate_im[ID_TOP:ID_BOT, ID_LEFT:ID_RIGHT]
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
        for j in contour:
            if (i % 2 == 0):
                x = contour[i]
                y = contour[i + 1]
                coords.append((x, y))

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

        coords = [list(tl), list(tr), list(bl), list(br)]

        return np.float32(coords).reshape(-1, 2)


def main(args):
    pp = PlatePull()

    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)