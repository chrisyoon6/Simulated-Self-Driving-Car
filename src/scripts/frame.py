#! /usr/bin/env python3

from __future__ import print_function

#import roslib; roslib.load_manifest('node')
import sys
import argparse
import os
import rospy
import cv2
import time
import math
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

# parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', required=True)

#Chris can you write this so that we can call from the command line '$ python3 frame.py X0-XX00' where X0 represents the parking spot number and 
# XX00 represents the license plate? Then we can easily capture data for license plate model and have the file name hold the info we need


class data_scraper:

    def __init__(self, img_name) -> None:
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)
        self.bridge = CvBridge()
        self.args = parser.parse_args()
        self.path = os.system('cd ../license-plate-data/' + self.args.name)
        
        # print("video writer opened: %b", self.video_writer.isOpened())


    def callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.path + '.png', frame)

def main(args):
  ds = data_scraper()
  rospy.init_node('controller', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

  #ds.video_writer.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)