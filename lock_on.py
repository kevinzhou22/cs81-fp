#!/usr/bin/env python

# Author: Muhtasim Miraz
# Date: 05/17/2023

import numpy as np
from sklearn.cluster import DBSCAN
import rospy # module for ROS APIs pylint: disable=import-error
from geometry_msgs.msg import PoseStamped # message type for cmd_vel pylint: disable=import-error
from std_msgs.msg import Float32

DEFAULT_OBJ_TOPIC = 'stalked' 
DEFAULT_OBJ_VEL_TOPIC = 'stalked_vel'
DEFAULT_ODOM_FRAME_ID = 'robot_0/odom'

# Frequency at which detect_object() is called
DETECT_FREQUENCY = 10

class MovingObjectDetector:
    def __init__(self, eps=1.5, min_samples=4, obj_topic=DEFAULT_OBJ_TOPIC, odom_frame_id=DEFAULT_ODOM_FRAME_ID, obj_vel_topic=DEFAULT_OBJ_VEL_TOPIC, detect_frequency=DETECT_FREQUENCY):
        """Constructor"""
        
        # maximum distance between two samples for them to be considered as in the same neighborhood
        # default is 1.5 grid units
        self.eps = eps

        # minimum number of samples in a neighborhood for a point to be considered as a core point
        # default is 4 points 
        self.min_samples = min_samples
        self.obj_topic = obj_topic
        self.odom_frame_id = odom_frame_id
        self.detect_frequency = detect_frequency
        
        # publisher for the object's pose
        self._obj_pub = rospy.Publisher(obj_topic, PoseStamped, queue_size=1)
        # publisher for the object's velocity
        self._vel_pub = rospy.Publisher(obj_vel_topic, Float32, queue_size=1)
        
        # keep track of the number of times detect_object() is called
        self._call_count = 0
        self._last_centroids = []
        self._last_times = []

    def detect_object(self, points):
        """Detects the object and publishes its pose and linear velocity to the stalked topic after detect_frequency number of calls to this function."""
        print('called')
        self._call_count += 1
        # proceed only if detect_object() has been called 10 times
        if self._call_count % self.detect_frequency != 0:
            return None
        
        points = np.array(points)
        # return None if points is an empty list
        if len(points) == 0:
            return None

        # cluster the points using DBSCAN
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
        labels = db.labels_
        clusters = [points[labels == i] for i in range(max(labels) + 1)]

        # filter out clusters that are too small
        clusters = [cluster for cluster in clusters if len(cluster) >= self.min_samples]
        print("clusters:\n", clusters)

        if not clusters:
            return None
        
        # find the largest cluster i.e. assume object is the largest cluster
        largest_cluster = max(clusters, key=lambda cluster: len(cluster))

        # compute the center of mass of that cluster
        centroid = np.mean(largest_cluster, axis=0)

        # save the last 3 centroids and times for theta and linear velocity calculation
        if len(self._last_centroids) == 3:
            self._last_centroids.pop(0)
            self._last_times.pop(0)
        self._last_centroids.append(centroid)
        self._last_times.append(rospy.get_rostime().to_sec())

        vector_1 = self._last_centroids[1] - self._last_centroids[0]
        vector_2 = self._last_centroids[2] - self._last_centroids[1]
        theta = np.arctan2(vector_2[1] - vector_1[1], vector_2[0] - vector_1[0])

        # publish the velocity of the object based on the last 3 centroids
        if len(self._last_centroids) == 3:
            velocities = [np.linalg.norm(self._last_centroids[i+1] - self._last_centroids[i]) / (self._last_times[i+1] - self._last_times[i]) for i in range(2)]
            average_velocity = np.mean(velocities)
            self._vel_pub.publish(Float32(average_velocity))

        pose = PoseStamped()
        pose.pose.position.x = centroid[0]
        pose.pose.position.y = centroid[1]
        pose.pose.orientation.z = theta
        pose.header.stamp = rospy.get_rostime()
        pose.header.frame_id = self.odom_frame_id
        self._obj_pub.publish(pose)

        return centroid