#! /usr/bin/env python3

from __future__ import print_function

#import roslib; roslib.load_manifest('node')
import sys
import rospy
import cv2
import os
from scrape_cmd import CmdScraper
from hsv_view import ImageProcessor
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import numpy as np

class DataScraper:
    SET_X = 0.5
    SET_Z = 1.0
    ERR_X = 0.1
    ERR_Z = 0.2
    WIDTH, HEIGHT = (1280, 720)
    FPS = 20
    COMPRESSION_RATIO = 0.25
    def __init__(self) -> None:
        """Creates a DataScraper object, repsonsible for scraping data from the simulation.
        """        
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback_img)
        self.twist_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.callback_twist)
        self.Twist = (0,0)

        self.bridge = CvBridge()
        self.dirPath_raw = "/home/fizzer/ros_ws/src/ENPH353-Team12/src/drive-data-raw/"
        self.dirPath_hsv = "/home/fizzer/ros_ws/src/ENPH353-Team12/src/drive-data-hsv/"
        self.count = 0

    def callback_img(self, data):
        """Callback for the subscriber node of the /image_raw topic.
        Saves both its raw and filtered images, labeling them with the latest Twist values.
        Ignores the input if the robot is not moving.

        Args:
            data (sensor_msgs::Image): The msg (image) recieved from /image_raw topic (i.e. robot's camera)
        """        
        if self.Twist[0] == 0 and self.Twist[1] == 0:
            return
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        frame = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
        print(self.Twist[0], self.Twist[1])
        x,z = DataScraper.discretize_vals(self.Twist[0], self.Twist[1], DataScraper.ERR_X, DataScraper.ERR_Z, DataScraper.SET_X, DataScraper.SET_Z)
        print(x, z)
        print('\n')
        self.save_raw(cv_image)
        name = "_".join([str(self.count), str(x), str(z)])
        name += ".png"
        cv2.imwrite(os.path.join(self.dirPath_raw, name), cv_image)
        hsv = ImageProcessor.filter(frame, ImageProcessor.white_low, ImageProcessor.white_up)
        cv2.imwrite(os.path.join(self.dirPath_hsv, "hsv_" + name), hsv)
        # cv2.imshow("hsv", lihsv)
        # cv2.resizeWindow("hsv", 300, 300)
        cv2.waitKey(1)
        self.count += 1
    def callback_twist(self, data):
        """Callback for the subscriber node of the /cmd_vel topic, called whenever there is a new message from this topic
        (i.e. new Twist values).

        Args:
            data (sensor_msgs::Twist): Twist object containing the robot's current velocities
        """        
        self.Twist = (data.linear.x, data.angular.z)
        # print(self.Twist[0], self.Twist[1])

    @staticmethod
    def compress(img, cmp_ratio):
        """Resizes the image using a compression ratio

        Args:
            img (cv::Mat): image to be compressed
            cmp_ratio (float): ratio to compress the image.

        Returns:
            cv::Mat: compressed image
        """        
        return cv2.resize(img, (0,0), fx=cmp_ratio, fy=cmp_ratio)
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
    def discretize_vals(x, z, err_x, err_z, set_x, set_z):
        """Discretize the input velocities (x,z) to the nearest of the following values:
        x = {0, set_x}
        z = {0, -set_z, set_z}

        Args:
            x (float): Input linear velocity value
            z (float): Input angular velocity value
            err_x (float): Tolerance for the input value 
            err_z (float): Tolerance for the input value 
            set_x (float): The discretized value
            set_z (float): The discretized value
        Raises:
            ValueError: If x or z velocities cannot be assigned to any of the above values.

        Returns:
            tuple[float,float]: the discretized x,z values
        """        
        new_x = x
        new_z = z
        if (x > set_x - err_x and x < set_x + err_x):
            new_x = set_x
        elif (x > -err_x and x < err_x):
            new_x = 0
        else:
            raise ValueError('x velocity cannot be discretized:', x)

        if (z > set_z - err_z and z < set_z + err_z):
            new_z = set_z
        elif (z > -1*set_z - err_z and z < -1*set_z + err_z):
            new_z = -1*set_z
        elif (z > -err_z and z < err_z):
            new_z = 0
        else:
            raise ValueError('z velocity cannot be discretized:', z)

        return (new_x, new_z)

def temp():
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    img = cv2.imread('/home/fizzer/ros_ws/src/ENPH353-Team12/src/drive-data-hsv/hsv_1_0.5_0.png')
    print(type(img), img.shape)
    # img = np.array(img)
    img = DataScraper.compress(img, 0.25)
    print(img.shape)
    img = DataScraper.crop(img, row_start=90)
    cv2.imshow('processed img', img)
    while cv2.waitKey(0) != ord('q'):
        pass
    cv2.destroyAllWindows()

from PIL import Image as Image_PIL
def temp2():
    folder = "/home/fizzer/ros_ws/src/ENPH353-Team12/src/drive-data-hsv"
    comp_folder = "/home/fizzer/ros_ws/src/ENPH353-Team12/src/drive-data-hsv-compressed"
    for filename in os.listdir(folder):
        img = np.array(Image_PIL.open(os.path.join(folder, filename)))
        img = DataScraper.compress(img, 0.25)
        img = DataScraper.crop(img, row_start=90)
        cv2.imwrite(os.path.join(comp_folder, filename), img)

def temp3():
    path = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/drive-data-hsv-compressed/hsv_0_0.5_0.png'
    path_orig = '/home/fizzer/ros_ws/src/ENPH353-Team12/src/drive-data-hsv/hsv_0_0.5_0.png'
    img = np.array(Image_PIL.open(path))
    img_orig = np.array(Image_PIL.open(path_orig))
    print(img.shape, img_orig.shape)

def main(args):
    ds = DataScraper()
    rospy.init_node('controller', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # main(sys.argv)
    # temp()
    temp2()
    temp3()
