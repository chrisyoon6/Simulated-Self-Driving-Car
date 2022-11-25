#! /usr/bin/env python3

from __future__ import print_function
from concurrent.futures import process

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class master:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw",Image,self.callback)

    self.vel_pub = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
    self.license_pub = rospy.Publisher("/license_plate", String, queue_size=1)

    self.out_plate = String()
    self.out_vel = Twist()
    self.counter = 0

    
  def callback(self,data):

    if (self.counter == 10):
        print(self.counter)
        self.license_pub.publish(String('TeamYoonifer,multi21,0,AA00'))
    elif (self.counter == 40):
        self.license_pub.publish(String('TeamYoonifer,multi21,-1,AA00'))
    else:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.out_vel.linear.x = 0
        self.out_vel.angular.z = -0.4

        self.vel_pub.publish(self.out_vel)

        cv2.waitKey(3)
    self.counter += 1



def main(args):
  rospy.init_node('master', anonymous=True)
  m = master()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)