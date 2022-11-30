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


class CmdScraper:

    def __init__(self) -> None:
        self.twist_sub = rospy.Subscriber("/R1/cmd_vel", Twist, self.callback_twist)
        self.image_sub = rospy.Subscriber("/R1/pi_camera/image_raw", Image, self.callback_img)
        self.Twist = (0, 0)

        self.cmd_vals = []

        self.bridge = CvBridge()

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self.video_writer = cv2.VideoWriter(
            'test_data.mp4', fourcc, 20, (1280, 720))

    def callback_img(self, data):
        if (self.Twist[0] != 0 or self.Twist[1] != 0):
            x = self.Twist[0]
            z = self.Twist[1]
            err = 0.1
            fwd = 0.4
            ang = 1.6
            # if (x > fwd-err and x < fwd+err and z > ang-4*err and z < ang+4*err):
            # 	x,z = (0.4, 1.6)
            # if (x > fwd-err and x < fwd+err and z > -ang-4*err and z < -ang+4*err):
            # 	x,z = (0.4, -1.6)
            # if (x > fwd-err and x < fwd+err and z > -4*err and z < 4*err):
            # 	x,z = (0.4, 0)
            # if (x > -err and x < err and z > ang-4*err and z < ang+4*err):
            # 	x,z = (0, 1.6)
            # if (x > -err and x < err and z > -ang-4*err and z < -ang+4*err):
            # 	x,z = (0, -1.6)
            x, z = CmdScraper.discretize_vals(x, z, err, 4*err, fwd, ang)
            self.cmd_vals.append((x, z))
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.video_writer.write(frame)

    def callback_twist(self, data):
        self.Twist = (data.linear.x, data.angular.z)

    @staticmethod
    def discretize_vals(x, z, err_x, err_z, set_x, set_z):
        '''
        Discretize the input values (x,z) the specified setpoint values (set_x, set_z) with a tolerance.
        Returns a tuple containing the discretized values.
        '''
        new_x = x
        new_z = z
        if (x > set_x - err_x and x < set_x + err_x):
            new_x = set_x
        else:
            new_x = 0

        if (z > set_z - err_z and z < set_z + err_z):
            new_z = set_z
        elif (z > -1*set_z - err_z and z < -1*set_z + err_z):
            new_z = -1*set_z
        else:
            new_z = 0

        return (new_x, new_z)


def main(args):
    ds = CmdScraper()
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
