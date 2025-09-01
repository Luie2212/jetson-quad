#!/usr/bin/env python3
"""
Python3 setup with catkin_virtualenv
"""

import rospy
import numpy as np
import cv2

if __name__ == '__main__':
    rospy.init_node("camera_loop_node")

    # wait until clock is initialised
    prevTime = 0
    while not prevTime:
        prevTime = rospy.Time.now()
    
    # # link camera
    
    # define resolution
    resCap = (2592, 1944)
    resWrite = resCap

    #open device and set capture resolution
    cap = cv2.VideoCapture("/dev/video1")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set( cv2.CAP_PROP_FRAME_WIDTH, resCap[0])
    cap.set( cv2.CAP_PROP_FRAME_HEIGHT, resCap[1])

    if not cap.isOpened():
        rospy.logerr("Error: Could not open camera.")
        exit()
    
    image_num=0

    # code below to loop
    rate = rospy.Rate(1) # rate to take a photo per second. Limit is ~20fps thorugh single image capture with the following implementation
    while not rospy.is_shutdown():
        
        # take screenshot:
        rospy.loginfo("Getting image...")
        ret, frame = cap.read()
        if ret:
            # Save the captured photo
            filename = f'image_{image_num:04x}.jpg'
            cv2.imwrite(f"/home/jetson/jetson-quad/Photos/{filename}", frame)
            rospy.loginfo(f"Saved Image as {filename}")
        else:
            rospy.loginfo("Failed to capture photo.")
            rospy.sleep(1)
            

        image_num += 1

        # call time
        currTime = rospy.Time.now()
        rospy.loginfo(f"Time elapsed since last call: {currTime.to_sec()-prevTime.to_sec():.4f}\n")
        prevTime = rospy.Time.now()
        
        rate.sleep()
    
    # cam.stop()
    cap.release()
    cv2.destroyAllWindows()