from geometry_msgs.msg import Twist
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import sys
import numpy as np

from hsv_view import ImageProcessor
from model import Model
from scrape_frames import DataScraper
from contour_approx import contour_approximator
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
    CROSSWALK_MSE_MOVING_THRES = 20
    DRIVE_PAST_CROSSWALK_FRAMES = 10 # 0.25s

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
        self.is_crossing_crosswalk = True

    def callback_img(self, data):
        """Callback function for the subscriber node for the /image_raw ros topic. 
        This callback is called when a new message has arrived to the /image_raw topic (i.e. a new frame from the camera).
        
        Args:
            data (sensor_msgs::Image): The image recieved from the robot's camera
        """        
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        '''
        if self.is_stopped_crosswalk:
            # in front of crosswalk, stopped.
            if self.can_cross_crosswalk(cv_image):
                self.is_stopped_crosswalk = False
                self.prev_mse_frame = None
                self.first_ped_stopped = False
                self.first_ped_moved = False
                self.is_crossing_crosswalk = True
            return
        
        if self.is_crossing_crosswalk:
            self.crossing_crosswalk_count += 1
            self.is_crossing_crosswalk = self.crossing_crosswalk_count < Driver.DRIVE_PAST_CROSSWALK_FRAMES

        if not self.is_crossing_crosswalk and self.is_red_line_close(cv_image):
            self.move.linear.x = 0
            self.move.angular.z = 0
            self.is_stopped_crosswalk = True

        # if driving: if close to red line, stop.
        # if stopped in front of red line: if pedestrian not moving (given some time), drive 
        '''
        hsv = DataScraper.process_img(cv_image)
        predicted = self.mod.predict(hsv)
        pred_ind = np.argmax(predicted)
        self.move.linear.x = Driver.ONE_HOT[pred_ind][0]
        self.move.angular.z = Driver.ONE_HOT[pred_ind][1]
        print(self.move.linear.x, self.move.angular.z)
        try:
            self.twist_pub.publish(self.move)
        except CvBridgeError as e: 
            print(e)
    
    def is_red_line_close(self, img):  
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)      
        red_filt = ImageProcessor.filter(hsv, ImageProcessor.red_low, ImageProcessor.red_up)
        cv2.imshow('script_view', red_filt)
        cv2.waitKey(3)
        area = contour_approximator.get_contours_area(red_filt,2)
        print("Area", area)
        return area[0] > Driver.CROSSWALK_FRONT_AREA_THRES and area[1] > Driver.CROSSWALK_BACK_AREA_THRES
    
    def can_cross_crosswalk(self, img): 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = ImageProcessor.crop(img_gray, 180, 720-180, 320, 1280-320)

        if self.prev_mse_frame is None:
            return False

        mse = ImageProcessor.compare_frames(self.prev_mse_frame, img_gray)
        
        self.prev_mse_frame = img_gray
        
        if mse < Driver.CROSSWALK_MSE_STOPPED_THRES:
            if not self.first_ped_stopped:
                self.first_ped_stopped = True
                return False
            if self.first_ped_moved and self.first_ped_stopped:
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
        print("Shutting down")
    cv2.destroyAllWindows()
    print("end")

if __name__ == '__main__':
    main(sys.argv)

