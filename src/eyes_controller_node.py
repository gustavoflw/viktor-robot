#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32

class AnglePublisherNode(object):
    def __init__(self):
        self.servo_angles = [90, 120, 60, 30, 150, 180]
        self.pub1 = rospy.Publisher('servo1', Int32, queue_size=1)
        self.pub2 = rospy.Publisher('servo2', Int32, queue_size=1)
        self.pub3 = rospy.Publisher('servo3', Int32, queue_size=1)
        self.pub4 = rospy.Publisher('servo4', Int32, queue_size=1)
        self.pub5 = rospy.Publisher('servo5', Int32, queue_size=1)
        self.pub6 = rospy.Publisher('servo6', Int32, queue_size=1)
        rospy.init_node('eyes_controller_node')

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.pub1.publish(self.servo_angles[0])
            self.pub2.publish(self.servo_angles[1])
            self.pub3.publish(self.servo_angles[2])
            self.pub4.publish(self.servo_angles[3])
            self.pub5.publish(self.servo_angles[4])
            self.pub6.publish(self.servo_angles[5])
            rate.sleep()

if __name__ == '__main__':
    try:
        node = AnglePublisherNode()
        node.run()
    except rospy.ROSInterruptException:
        pass