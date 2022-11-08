#! /usr/bin/env python3

from __future__ import print_function

#import roslib; roslib.load_manifest('node')
import sys
import rospy
import cv2
import time
import math
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

class data_scraper:

    def __init__(self) -> None:
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
        self.bridge = CvBridge()
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_writer = cv2.VideoWriter('test_data.mp4', fourcc, 20, (1280, 720))
        # print("video writer opened: %b", self.video_writer.isOpened())


    def callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.video_writer.write(frame)


def main(args):
  ds = data_scraper()
  rospy.init_node('controller', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

  ds.video_writer.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)