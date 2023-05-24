#!/usr/bin/env python

# Author: Amel Docena
# Date: May 23, 2023


"""
Generate training data:
X data
1. State = f(robot pose, object pose) PO: relative velocity of robot and object. If such, then robot velocity should be published
2. Action = linear velocity, angular velocity, duration

Y data
1. (Resultant) State = f(robot pose, object pose) after taking action

TODO: The robot and object poses should be in the same ref frame.
PO1: Transform the object pose wrt the robot's odom frame
PO2: Transform robot and object poses wrt map frame
"""

import os
import numpy as np
import tf
import rospy
import math
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from reset_simulation import *

ROBOT_ODOM_TOPIC = '/robot_0/odom'
ROBOT_CMD_TOPIC = '/robot_0/cmd_vel'
OBJECT_ODOM_TOPIC = '/robot_1/odom'

class GenerateTrainData:
    def __init__(self, node_name, robot_odom_topic=ROBOT_ODOM_TOPIC, robot_cmd_topic=ROBOT_CMD_TOPIC, object_odom_topic=OBJECT_ODOM_TOPIC):
        rospy.init_node(node_name)

        #Subscribers and Publishers
        #Subscribe to robot odom
        rospy.Subscriber(robot_odom_topic, Odometry, self._set_robot_pose_cb, queue_size=1)

        #PO: Subscribe to object's pose
        rospy.Subscriber(object_odom_topic, Odometry, self._set_object_pose_cb, queue_size=1)

        #Publish cmd_vel to robot
        self._robot_cmd_pub = rospy.Publisher(robot_cmd_topic, Twist, queue_size=1)

        #Parameters and variables
        self.curr_robot_pose = None
        self.curr_object_pose = None

    def _set_robot_pose_cb(self, msg):
        #Sets curr robot pose
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.curr_robot_pose = (x, y, theta)
        #print("Curr robot pose:", self.curr_robot_pose)

    def _set_object_pose_cb(self, msg):
        #Sets curr object pose
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.curr_object_pose = (x, y, theta)
        #print("Curr object pose:", self.curr_object_pose)

    def get_curr_robot_pose(self):
        #Returns current robot pose
        return self.curr_robot_pose

    def get_curr_object_pose(self):
        #Returns current object pose
        return self.curr_object_pose

    def get_curr_state(self):
        """
        Gets current state: robot pose and object pose
        Returns:

        """
        robot_pose = self.get_curr_robot_pose()
        object_pose = self.get_curr_object_pose()
        curr_state = [robot_pose, object_pose]

        return curr_state

    def move_robot(self, linear_vel, angular_vel, duration):
        """Send a velocity command (linear vel in m/s, angular vel in rad/s) for a given duration."""
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        start_time = rospy.get_rostime()

        while rospy.get_rostime() - start_time <= rospy.Duration(duration):
            self._robot_cmd_pub.publish(twist_msg)

    def get_next_state(self, action):
        """
        We get next state, i.e., pose of robot relative to object after taking an action

        Returns:

        """
        linear_vel, angular_vel, duration = action

        #Publish to robot 1 cmd vel to take action
        self.move_robot(linear_vel, angular_vel, duration)

        #Get the robot pose and object pose after taking action
        robot_pose = self.get_curr_robot_pose()
        object_pose = self.get_curr_object_pose()
        next_state = [robot_pose, object_pose]

        return next_state

    def generate_train_data(self, range_action_dict):
        """
        Generates train data
        Returns:

        """
        #TODO: We can generate from different starting current states too.
        curr_state = self.get_curr_state()

        #PO: Loop for action increment
        v_range = np.linspace(range_action_dict['v'][0], range_action_dict['v'][1], range_action_dict['v'][2])
        w_range = np.linspace(range_action_dict['w'][0], range_action_dict['w'][1], range_action_dict['w'][2])
        t_range = np.linspace(range_action_dict['t'][0], range_action_dict['t'][1], range_action_dict['t'][2])

        data = list()
        for v in v_range:
            for w in w_range:
                for t in t_range:
                    action = (v, w, t)
                    print("Action:", action)
                    next_state = self.get_next_state(action) #Get next state: resultant robot pose, object pose
                    data_pt = (curr_state, next_state) #Store in data
                    data.append(data_pt)
                    #reset_simulation() #TODO: Reset simulation
                    rospy.sleep(1)
        print("Data:", len(data), data)

    """
    PO Sanity check:
    The robot should move at each iteration/action sent. DONE
    We now do the resetting of simulation
    
    
    """

if __name__ == "__main__":
    #Range per action: (tuple) start, stop, number of evenly-spaced data points
    range_action_dict = {'v': (0, 1.0, 5),
                         'w': (-math.pi, math.pi, 5),
                         't': (0, 2.0, 5)}

    train_data_generator = GenerateTrainData('gen_train_data')
    train_data_generator.generate_train_data(range_action_dict)