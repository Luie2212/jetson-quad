#!/usr/bin/env python3
"""
Python3 setup with catkin_virtualenv
"""

import rospy
import numpy as np
import pygame
import pygame.camera

if __name__ == '__main__':
    rospy.init_node("camera_loop_node")

    # wait until clock is initialised
    prevTime = 0
    while not prevTime:
        prevTime = rospy.Time.now()
    
    # link camera
    pygame.camera.init()
    cameras_found = len(pygame.camera.list_cameras())
    if cameras_found == 0: #Camera detected or not
        rospy.logerr("Camera list is empty! Exiting...")
        exit()
    rospy.loginfo(f"Cameras found: {cameras_found}")
    cam = pygame.camera.Camera("/dev/video0",(1920,1080))
    cam.start()
    image_num=0

    # code below to loop
    rate = rospy.Rate(1) # rate to take a photo per second. Limit is ~5fps thorugh single image capture with the following implementation
    while not rospy.is_shutdown():
        
        # take screenshot:
        rospy.loginfo("Getting image...")
        while not cam.query_image():
            rospy.loginfo("Waiting for cam to be ready...")
            rospy.sleep(1)
        img = cam.get_image()
        rospy.loginfo("...Image recieved. Saving Image...")
        filename = f'image_{image_num:04x}.jpg'
        pygame.image.save(img,f"/home/jetson/jetson-quad/Photos/{filename}")
        rospy.loginfo(f"Saved Image as {filename}")

        image_num += 1

        # call time
        currTime = rospy.Time.now()
        rospy.loginfo(f"Time elapsed since last call: {currTime.to_sec()-prevTime.to_sec():.4f}\n")
        prevTime = rospy.Time.now()
        
        rate.sleep()
    
    cam.stop()