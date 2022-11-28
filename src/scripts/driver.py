import rospy
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import sys
import numpy as np

from hsv_view import ImageProcessor
import model
from scrape_frames import DataScraper

class Driver:
    DEF_VALS = (0.5, 0.5)
    MODEL_PATH = "/home/fizzer/ros_ws/src/models/drive_model1.h5"
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

    def __init__(self):
        """Creates a Driver object. 
        """            
        self.twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback_img)
        self.move = Twist()
        self.bridge = CvBridge()

        self.move.linear.x = 0
        self.move.angular.z = 0

        self.mod = model.Model(Driver.MODEL_PATH)
    
    def callback_img(self, data):
        """Callback function for the subscriber node for the /image_raw ros topic. 
        This callback is called a new message has arrived to the /image_raw topic (i.e. a new frame from the camera).
        
        Args:
            data (sensor_msgs::Image): The image recieved from the robot's camera
        """        
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        frame = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
        hsv = ImageProcessor.filter(frame, ImageProcessor.white_low, ImageProcessor.white_up)
        predicted = self.mod.predict(hsv)
        pred_ind = np.argmax(predicted)
        self.move.linear.x = Driver.ONE_HOT[pred_ind][0]
        self.move.angular.z = Driver.ONE_HOT[pred_ind][1]
        try:
            self.twist_pub.publish(self.move)
        except CvBridgeError as e: 
            print(e)


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

