#! /usr/bin/env python3

from __future__ import print_function

import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class Master:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            "/R1/pi_camera/image_raw", Image, self.callback)

        self.move = rospy.Publisher("/R1/cmd_vel", Twist, queue_size=1)
    
        self.out_vel = Twist()
        self.counter = 0
        self.inner_counter = 0
        self.inner_bool = False

        self.turning_seq_counter1 = 0
        self.turning_transition = False

    def callback(self, data):  ### TO BE REWRITTEN

        if not self.inner_bool and not self.turning_transition:
            self.inner_bool = True

        if self.turning_transition:
            if (self.inner_counter < 12):
                self.out_vel.linear.x = 0.8
                self.out_vel.angular.z = 1.6
            elif (self.inner_counter < 18):
                self.out_vel.linear.x = 0
                self.out_vel.angular.z = 1.2
            else:
                self.out_vel.linear.x = 0
                self.out_vel.angular.z = 0
                self.inner_bool = True
                self.turning_transition = False
            self.move.publish(self.out_vel)
            print(self.inner_counter)

            self.inner_counter += 1
            return

        if self.inner_bool:
            if (self.turning_seq_counter1 < 20):
                self.out_vel.linear.x = 0
                self.out_vel.angular.z = 1.54
            else:
                self.out_vel.angular.z = 0
                self.out_vel.linear.x = 0
                
                self.inner_bool = False
                
                ###################### CHECK FOR CAR
                # check_for_car = True
                
                self.turning_transition = True

            self.move.publish(self.out_vel)
            self.turning_seq_counter1 += 1
            return

def main(args):
    rospy.init_node('master', anonymous=True)
    m = Master()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
