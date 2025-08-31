#!/usr/bin/env python
"""
Python3 setup with catkin_virtualenv
"""

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# global image number variable
image_n = 0; 

def callback(data, startTime):
    global image_n

    # take screenshot:
    rospy.loginfo("Getting image...")
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    
    # Save the captured photo
    filename = "image_%04x.jpg" % image_n
    cv2.imwrite("/home/jetson/jetson-quad/Photos/%s" % filename, cv_image)
    rospy.loginfo("Saved Image as %s" % filename)

    # increment image n
    image_n += 1




if __name__ == '__main__':
    
    # define subscriber and node
    rospy.init_node("camera_receive")

    # wait until clock is initialised
    startTime = 0
    while not startTime:
        startTime = rospy.Time.now()

    sub = rospy.Subscriber('camera_reading', Image, callback, callback_args=(startTime))

    # spin node
    rospy.spin();