#!/usr/bin/python
"""
Converts mavros readings into transform messages usable by Sunshine

At the moment, this only works using a manually computed altitude
"""

from sensor_msgs.msg import NavSatFix, PointCloud2, Image
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Vector3, Quaternion
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64
import rospy
import navpy
import cv2
import tf

TANK_WATER_HEIGHT = 1.346 # meters

class Mav2Sunshine():
    def __init__(self):
        self.home_lat = 40
        self.home_lon = -71
        self.home_alt = 0

        self.ts = TransformStamped()
        self.img = Image()
        self.img_depth = Image()
        self.pcl = PointCloud2()
        self.depth = 0
        self.alt = 0

        self.use_img_depth = False
        
        #rospy.Subscriber('mavros/underwater_gps/fix', NavSatFix, self.global_pos_cb)
        rospy.Subscriber('mavros/global_position/global', NavSatFix, self.global_pos_cb)
        rospy.Subscriber('mavros/global_position/rel_alt', Float64, self.altitude_cb)
        rospy.Subscriber('mavros/global_position/local', Odometry, self.local_pos_cb)

        self.transform_pub = rospy.Publisher('/camera_world_transform', TransformStamped, queue_size=1)

    def global_pos_cb(self, sat_msg):
        # navsatfix
        ned = navpy.lla2ned(sat_msg.latitude, sat_msg.longitude, -self.depth, self.home_lat, self.home_lon, self.home_alt)
        self.ts.header = sat_msg.header
        self.ts.header.frame_id = "map"

        # ENU coordinates returned
        self.ts.transform.translation.x = ned[1]
        self.ts.transform.translation.y = ned[0]
        self.ts.transform.translation.z = -ned[2]

        br = tf.TransformBroadcaster()
        br.sendTransform((ned[1], ned[0], -ned[2]), (self.ts.transform.rotation.x, self.ts.transform.rotation.y, self.ts.transform.rotation.z, self.ts.transform.rotation.w), rospy.Time.now(), self.ts.child_frame_id, self.ts.header.frame_id)

        # Orientation captured from before
        self.transform_pub.publish(self.ts)

    def altitude_cb(self, alt_msg):
        self.depth = alt_msg.data
        self.alt = TANK_WATER_HEIGHT - self.depth
        
    def local_pos_cb(self, odom_msg):
        # odom
        self.ts.child_frame_id = "zed_left_camera_frame" # odom_msg.child_frame_id
        self.ts.transform.rotation = odom_msg.pose.pose.orientation


if __name__ == '__main__':
    rospy.init_node('mav2tf')
    mav2s = Mav2Sunshine()
    rospy.spin()
