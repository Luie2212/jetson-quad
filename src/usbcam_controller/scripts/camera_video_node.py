#!/usr/bin/env python3
"""
Python3 setup with catkin_virtualenv
"""

import rospy
import numpy as np
import cv2

if __name__ == '__main__':
    rospy.init_node("camera_video_node")

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
    
    #settup video codec  and writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for AVI
    out = cv2.VideoWriter('/home/jetson/jetson-quad/Photos/Videos/output.mp4', fourcc, 15.0, resCap) # 15 FPS, resCap resolution

    # code below to loop
    while not rospy.is_shutdown():          
        # take screenshot:
        rospy.loginfo("Getting image...")
        ret, frame = cap.read()
        if ret:
            # Save the captured frame to video file
            out.write(frame)
        else:
            rospy.loginfo("Failed to capture frame. Frame dropped.")
            

        # call time
        currTime = rospy.Time.now()
        rospy.loginfo(f"Time elapsed since last call: {currTime.to_sec()-prevTime.to_sec():.4f}\n")
        prevTime = rospy.Time.now()
            
    # cam.stop()
    cap.release()
    cv2.destroyAllWindows()