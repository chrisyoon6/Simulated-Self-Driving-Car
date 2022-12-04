from geometry_msgs.msg import Twist
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import sys
import numpy as np
from time import sleep

from hsv_view import ImageProcessor
from model import Model
from scrape_frames import DataScraper
from plate_reader import PlateReader
from pull_plate import PlatePull
from copy import deepcopy
class Driver:
    DEF_VALS = (0.5, 0.5)
    MODEL_PATH = "/home/fizzer/ros_ws/src/models/drive_model-0.h5"
    """
    (0.5,0) = 0
    (0, -1) = 1
    (0, 1) = 2
    (0.5, -1) = 3
    (0.5, 1) = 4
    """
    ONE_HOT = { 
        0 : (DataScraper.SET_X, 0),
        1 : (0, -1*DataScraper.SET_Z),
        2 : (0, DataScraper.SET_Z),
        3 : (DataScraper.SET_X, -1*DataScraper.SET_Z),
        4 : (DataScraper.SET_X, DataScraper.SET_Z)
    }
    CROSSWALK_FRONT_AREA_THRES = 8000
    CROSSWALK_BACK_AREA_THRES = 500
    FPS = 20
    CROSSWALK_MSE_STOPPED_THRES = 8
    CROSSWALK_MSE_MOVING_THRES = 40
    DRIVE_PAST_CROSSWALK_FRAMES = int(FPS*10)
    FIRST_STOP_SECS = 2
    
    PLATE_SECTION_SECS = 3
    PLATE_DRIVE_BACK_SECS = 1

    PLATE_SEC_X = 0.1
    PLATE_SEC_Z = 0.2

    ROWS = 720
    COLS = 1280
    
    def __init__(self):
        """Creates a Driver object. Responsible for driving the robot throughout the track. 
        """            
        self.twist_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback_img)
        self.move = Twist()
        self.bridge = CvBridge()

        self.move.linear.x = 0
        self.move.angular.z = 0

        self.mod = Model(Driver.MODEL_PATH)

        self.is_stopped_crosswalk = False
        self.first_ped_moved = False
        self.first_ped_stopped = False
        self.prev_mse_frame = None
        self.crossing_crosswalk_count = 0
        self.is_crossing_crosswalk = False
        self.first_stopped_frames_count = 0

        self.pr = PlateReader(script_run=False)
        self.lp = ""

        self.at_plate = False
        self.plate_frames_count = 0
        self.plate_drive_back = False
        self.drive_back_frames_count = 0

    def callback_img(self, data):
        """Callback function for the subscriber node for the /image_raw ros topic. 
        This callback is called when a new message has arrived to the /image_raw topic (i.e. a new frame from the camera).
        Using the image, it conducts the following:

        1) drives and looks for a red line (if not crossing the crosswalk)
        2) if a red line is seen, stops the robot
        3) drives past the red line when pedestrian is not crossing
        
        Args:
            data (sensor_msgs::Image): The image recieved from the robot's camera
        """        
        # cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        if self.is_stopped_crosswalk:
            # print("stopped crosswalk")
            if self.can_cross_crosswalk(cv_image):
                # print("can cross")
                self.is_stopped_crosswalk = False
                self.prev_mse_frame = None
                self.first_ped_stopped = False
                self.first_ped_moved = False
                self.is_crossing_crosswalk = True
            return

        # print("driving")

        if self.is_crossing_crosswalk:
            # print("crossing")
            self.crossing_crosswalk_count += 1
            self.is_crossing_crosswalk = self.crossing_crosswalk_count < Driver.DRIVE_PAST_CROSSWALK_FRAMES

        hsv = DataScraper.process_img(cv_image, type="bgr")

        predicted = self.mod.predict(hsv)
        pred_ind = np.argmax(predicted)
        self.move.linear.x = Driver.ONE_HOT[pred_ind][0]
        self.move.angular.z = Driver.ONE_HOT[pred_ind][1]

        print(self.move.linear.x, self.move.angular.z)

        # check if red line close only when not crossing
        if not self.is_crossing_crosswalk and self.is_red_line_close(cv_image):
            self.crossing_crosswalk_count = 0 
            # print("checking for red line")
            self.move.linear.x = 0.0
            self.move.angular.z = 0.0
            self.is_stopped_crosswalk = True
            self.first_stopped_frame = True
    
        if self.plate_drive_back:
            print("driving back")
            self.drive_back_frames_count += 1
            self.plate_drive_back = self.drive_back_frames_count < int(Driver.FPS*Driver.PLATE_DRIVE_BACK_SECS)
            self.move.linear.x = -1*DataScraper.SET_X/2
            self.move.angular.z = 0
            # self.twist_pub.publish(self.move)
            return
        else:
            self.drive_back_frames_count = 0

        r_st = int(Driver.ROWS/2.5)
        r_en = -1
        c_st = Driver.COLS // 5
        c_en = Driver.COLS // 5*4
        if self.at_plate:
            c_st = -1
            c_en = -1
        # crped = ImageProcessor.crop(cv_image, r_st, r_en, c_st, c_en)
        crped = ImageProcessor.crop(cv_image, r_st, r_en, -1, -1)
        blu_crped = ImageProcessor.filter(crped, ImageProcessor.blue_low, ImageProcessor.blue_up)
        cv2.imshow("Cropped", blu_crped)
        cv2.waitKey(1)
        print("-------Blue area: ", PlatePull.get_contours_area(blu_crped))
        lp = self.pr.get_license_plate(crped)

        if not self.at_plate and lp:
            # first time at plate section
            print("First plate")
            self.at_plate = True
            self.plate_drive_back = True
            return
        elif not self.at_plate:
            print("not at plate")
            self.plate_frames_count = 0
        
        if self.at_plate:
            print("at plate")
            # at plate location
            x = round(self.move.linear.x,1)
            z = round(self.move.angular.z, 1)
            if x == DataScraper.SET_X:
                self.move.linear.x /= 10.0
            if z == DataScraper.SET_Z or z == -1*DataScraper.SET_Z:
                self.move.angular.z /= 10.0
            
            self.at_plate = self.plate_frames_count < int(Driver.PLATE_SECTION_SECS*Driver.FPS)
            self.plate_frames_count += 1

        if self.lp:
            print(self.lp)
        try:
            print("------------Move-------------",self.move.linear.x, self.move.angular.z)
            self.twist_pub.publish(self.move)
            pass
        except CvBridgeError as e: 
            print(e)
    
    def is_red_line_close(self, img):  
        """Determines whether or not the robot is close to the red line.

        Args:
            img (cv::Mat): The raw RGB image data to check if there is a red line

        Returns:
            bool: True if deemed close to the red line, False otherwise.
        """        
        red_filt = ImageProcessor.filter(img, ImageProcessor.red_low, ImageProcessor.red_up)
        # cv2.imshow('script_view', red_filt)
        # cv2.waitKey(3)
        area = PlatePull.get_contours_area(red_filt,2)
        # print("Area", area)
        if not list(area):
            return False

        return area[0] > Driver.CROSSWALK_FRONT_AREA_THRES and area[1] > Driver.CROSSWALK_BACK_AREA_THRES
    
    def can_cross_crosswalk(self, img): 
        """Determines whether or not the robot can drive past the crosswalk. Only to be called when 
        it is stopped in front of the red line. 
        Updates this object.

        Can cross if the following conditions are met:
        - First the robot has stopped for a sufficient amount of time to account for stable field of view 
        due to inertia when braking
        - Robot must see the pedestrian move across the street at least once
        - Robot must see the pedestrian stopped at least once
        - Robot must see the pedestrian to be in a stopped state.

        Args:
            img (cv::Mat): Raw RGB iamge data

        Returns:
            bool: True if the robot able to cross crosswalk, False otherwise
        """        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = ImageProcessor.crop(img_gray, 180, 720-180, 320, 1280-320)

        if self.prev_mse_frame is None:
            self.prev_mse_frame = img_gray
            return False
        
        
        mse = ImageProcessor.compare_frames(self.prev_mse_frame, img_gray)
        # print("mse:", mse)
        # print("first ped stopped, first ped move:" , self.first_ped_stopped, self.first_ped_moved)
        self.prev_mse_frame = img_gray
        
        if self.first_stopped_frames_count <= int(Driver.FIRST_STOP_SECS*Driver.FPS):
            self.first_stopped_frames_count += 1
            return False

        if mse < Driver.CROSSWALK_MSE_STOPPED_THRES:
            if not self.first_ped_stopped:
                self.first_ped_stopped = True
                return False
            if self.first_ped_moved and self.first_ped_stopped:
                self.prev_mse_frame = None
                self.first_stopped_frames_count = 0
                return True

        if mse > Driver.CROSSWALK_MSE_MOVING_THRES:
            if not self.first_ped_moved:
                self.first_ped_moved = True
                return False

        return False
        

def main(args):    
    rospy.init_node('Driver', anonymous=True)
    dv = Driver()
    try:
        rospy.spin()
    except KeyboardInterrupt:
            ("Shutting down")
    cv2.destroyAllWindows()
    print("end")

if __name__ == '__main__':
    main(sys.argv)

