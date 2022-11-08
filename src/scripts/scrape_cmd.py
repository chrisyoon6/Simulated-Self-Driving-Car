#! /usr/bin/env python3

from __future__ import print_function

#import roslib; roslib.load_manifest('node')
import sys
import rospy
import cv2
import time
import csv
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

class data_scraper:

    def __init__(self) -> None:
        self.twist_sub = rospy.Subscriber("/R1/cmd_vel",Twist,self.callback_twist)
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback_img)
        self.Twist = (0,0)
        
        self.cmd_vals = []


    def callback_img(self, data):
        x = self.Twist[0]
        z = self.Twist[1]
        self.cmd_vals.append((x,z))

    def callback_twist(self,data):
        self.Twist = (data.linear.x, data.angular.z)


def main(args):
  ds = data_scraper()
  rospy.init_node('controller', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

  with open('text_data.csv', 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerows(ds.cmd_vals)

  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)