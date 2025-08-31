#!/usr/bin/env python3
"""
Python3 setup with catkin_virtualenv
"""

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

if __name__ == '__main__':
    
    # define publisher and node
    pub = rospy.Publisher('camera_reading', Image, queue_size=10)
    rospy.init_node("camera_publish")

    # wait until clock is initialised
    prevTime = 0
    while not prevTime:
        prevTime = rospy.Time.now()
    
    # # link camera
    
    # define resolution
    resCap = (2592, 1944)
    resWrite = resCap

    #open device and set capture resolution
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set( cv2.CAP_PROP_FRAME_WIDTH, resCap[0])
    cap.set( cv2.CAP_PROP_FRAME_HEIGHT, resCap[1])

    if not cap.isOpened():
        rospy.logerr("Error: Could not open camera.")
        exit()
    
    # code below to loop
    rate = rospy.Rate(1) # rate to take a photo per second. Limit is ~20fps thorugh single image capture with the following implementation
    while not rospy.is_shutdown():
        
        # take screenshot:
        rospy.loginfo("Getting image...")
        ret, frame = cap.read()

        # call time
        currTime = rospy.Time.now()
        rospy.loginfo(f"Time elapsed since last call: {currTime.to_sec()-prevTime.to_sec():.4f}\n")
        prevTime = rospy.Time.now()

        # publish to /camera_reading
        bridge = CvBridge()
        ros_image_msg = bridge.cv2_to_imgmsg(frame, encoding="passthrough")
        pub.publish(ros_image_msg)

        rate.sleep()
    
    # cam.stop()
    cap.release()
    cv2.destroyAllWindows()