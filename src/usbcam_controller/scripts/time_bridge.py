#!/usr/bin/env python
import rospy
from sensor_msgs.msg import NavSatFix

# Path to the temporary file that Python 3.8 YOLO script will read
TIME_FILE = "/tmp/ros_time.txt"

def callback(msg):
    """Write ROS timestamp to a file as secs.nsecs"""
    try:
        with open(TIME_FILE, "w") as f:
            print("Found message: %d.%d\n" % (msg.header.stamp.secs, msg.header.stamp.nsecs))
            f.write("%d.%d\n" % (msg.header.stamp.secs, msg.header.stamp.nsecs))
    except Exception as e:
        rospy.logwarn("Failed to write ROS time: %s" % e)

if __name__ == "__main__":
    rospy.init_node('time_bridge_file')
    rospy.Subscriber("/mavros/global_position/global", NavSatFix, callback)
    rospy.loginfo("Time bridge node started, writing ROS time to %s" % TIME_FILE)
    rospy.spin()