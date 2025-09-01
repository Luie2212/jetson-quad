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
    resCap = (1920, 1080)
    fps = 30
    resWrite = resCap

    #open device and set capture resolution
    cap = cv2.VideoCapture("/dev/video1")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set( cv2.CAP_PROP_FRAME_WIDTH, resCap[0])
    cap.set( cv2.CAP_PROP_FRAME_HEIGHT, resCap[1])
    cap.set(cv2.CAP_PROP_FPS, fps) # set fps to desired quantity. Issue currently with matching this fps due to possibly maxing out cpu thread

    if not cap.isOpened():
        rospy.logerr("Error: Could not open camera.")
        exit()

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])
    print("Codec in use:", codec)
    print("FPS reported:", cap.get(cv2.CAP_PROP_FPS))
    print("Resolution:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #settup video codec  and writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Codec 
    out = cv2.VideoWriter('/home/jetson/jetson-quad/Photos/Videos/output.avi', fourcc, fps, resCap) # 30 FPS, resCap resolution
    
    # Test if ssd is bottleneck
    # out = cv2.VideoWriter('/dev/shm/output.avi', fourcc, fps, resCap) # 30 FPS, resCap resolution

    # timing trackers
    prevTime = rospy.Time.now()
    windowStart = prevTime
    intervals = []

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
        # currTime = rospy.Time.now()
        # rospy.loginfo(f"Time elapsed since last call: {currTime.to_sec()-prevTime.to_sec():.4f}\n")
        # prevTime = rospy.Time.now()
            
        # measure interval
        currTime = rospy.Time.now()
        dt = currTime.to_sec() - prevTime.to_sec()
        intervals.append(dt)
        prevTime = currTime
            
            # once per second, report average delay
        if (currTime - windowStart).to_sec() >= 1.0:
            if intervals:
                avg_dt = sum(intervals) / len(intervals)
                fps_measured = 1.0 / avg_dt if avg_dt > 0 else 0
                rospy.loginfo(
                    f"Average frame interval: {avg_dt:.4f} s "
                    f"({fps_measured:.2f} fps) over last {len(intervals)} frames"
                )
            # reset for next window
            intervals.clear()
            windowStart = currTime
            
    # cam.stop()
    cap.release()
    out.release()
    cv2.destroyAllWindows()