#!/usr/bin/env python

# python/class libraries
import numpy as np
import q_learning
import math

# rospy libraries
import rospy
from geometry_msgs.msg import Twist, PoseStamped # message type
from nav_msgs.msg import OccupancyGrid
import tf

# CONSTANTS
# Frequency at which the loop operates
VELOCITY = 0.3 #m/s
ANG_VELOCITY = math.pi/4.0 #rad/s
DEFAULT_SCAN_TOPIC = 'base_scan'
DEFAULT_OBJ_TOPIC = 'stalked'

class Grid:
    """Initializes a Grid class which stores height, width, resolution data."""

    def __init__(self, occupancy_grid_data, width, height, resolution):
        self.grid = np.reshape(occupancy_grid_data, (height, width))
        self.resolution = resolution

class qMove:
    def __init__(self):
        """Initialization."""
        self.sub = rospy.Subscriber("map", OccupancyGrid, self.map_callback, queue_size=1)
        self.publisher = rospy.Publisher("/robot_0/cmd_vel", Twist, queue_size=1)
        self.vel_msg = Twist()
        self.rate = rospy.Rate(10)
        self.q_policy = None # table holding policies at different states
        self.map = None # the variable containing the map.
        self.height = None # height of the map
        self.width = None # width of the map
        self.curr_x = None # holds the current x position
        self.curr_y = None # holds the current y position
        self.curr_angle = None # holds the current angle
        self.target_loc = None
        self._obj_sub = rospy.Subscriber(DEFAULT_OBJ_TOPIC, PoseStamped, self._obj_callback, queue_size=1) 
        self.map_T_odom = np.array([[1, 0, 0, 5],
                                    [0, 1, 0, 5],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        self.transform_listener = tf.TransformListener()
        self.done_travelling = False
    
    def _obj_callback(self, msg):
        self.done_travelling = False
        
        # Find the position of the robot in odom reference frame
        odom_T_bl = self._get_transformation_matrix('robot_0/odom', 'robot_0/base_link')
        yaw = self._get_rotation('robot_0/base_link', 'robot_0/odom')
        robot_loc_odom = self._transform_2d_point(odom_T_bl, ((0, 0)))

        # Find the position of the robot in map reference frame
        robot_loc_map = self._transform_2d_point(self.map_T_odom, robot_loc_odom)
        self.curr_x = int(round(robot_loc_map[0]))
        self.curr_y = int(round(robot_loc_map[1]))
        self.curr_angle = yaw
        
        # position of stalker in map
        stalker_map = self._transform_2d_point(self.map_T_odom, ((msg.pose.position.x, msg.pose.position.y)))
        self.target_loc = (int(round(stalker_map[0])), int(round(stalker_map[1])))

        while (self.done_travelling != True):
            print("Targeted Location Locked!")
            print(self.target_loc)
            self.get_table()
            self.done_travelling = self.follow_policy()

    
    def _get_transformation_matrix(self, target, source, time=rospy.Time(0)):
        """Gets the transformation matrix from target to source, looping until found"""
        has_found_transformation = False
        # it's possible the relevant transformation has not been published yet
        # this just loops until the transformation is acquired
        while not has_found_transformation:
            try:
                (trans, rot) = self.transform_listener.lookupTransform(
                    target,
                    source, 
                    time
                )
                has_found_transformation = True
            except (tf.LookupException, tf.ExtrapolationException):
                rospy.sleep(0.1)
                continue

        t = tf.transformations.translation_matrix(trans)
        R = tf.transformations.quaternion_matrix(rot)
        transformation_matrix = np.matmul(t, R)
        return transformation_matrix
    
    
    def _transform_2d_point(self, transformation_matrix, point):
        """
        transforms the inputted point to the reference frame given
        by the transformation matrix
        """
        x, y = point
        transformed_point = np.matmul(transformation_matrix, np.array([x, y, 0, 1], dtype='float64'))
        return (transformed_point[0], transformed_point[1])
    
    def _get_rotation(self, target, source):
        """
        Gets the angle in the pose of the robot in the target
        w.r.t the source reference frame
        """
        has_found_transformation = False
        while not has_found_transformation:
            try:
                trans, rot = self.transform_listener.lookupTransform(
                    source,
                    target,
                    rospy.Time(0)
                )
                has_found_transformation = True
            except tf.LookupException:
                rospy.sleep(0.5)
                continue
        yaw = tf.transformations.euler_from_quaternion(rot)[2]
        return yaw
        

    def map_callback(self, msg):
        """Initializes map as Grid object; calls method to get best policy"""
        # create grid object upon receipt of map
        self.map = Grid(msg.data, msg.info.width, msg.info.height, msg.info.resolution)
        self.height  = int(msg.info.height*msg.info.resolution)
        self.width = int(msg.info.width*msg.info.resolution)

        # if policy has not been created; generate table
        if self.q_policy == None:
            if self.target_loc != None:
                self.get_table()

    def get_table(self):
        """Handles Q-learning; sets q_policy instance variable to policy found after training"""
        # check to see if map has been read
        if self.map != None:
            # start the q-learning
            q_model = q_learning.QLearning(width=self.width, height=self.height, start_loc=(self.curr_x, self.curr_y), target_loc=self.target_loc)
            q_model.training(10) # training for 5 iterations
            q_model.get_best_policy(q_model.q_table)
            self.q_policy = q_model.best_policy # setting the best policy
        
    def move(self, linear_x, angular_z, start_time, duration):
        """moves robot by a specified velocity for a specified duration"""
        self.vel_msg.linear.x = linear_x
        self.vel_msg.angular.z = angular_z
        while not rospy.is_shutdown():
            if rospy.get_rostime() - start_time >= rospy.Duration(duration):
                break
            self.publisher.publish(self.vel_msg)
            self.rate.sleep()
    
    def follow_policy(self):
        """Controls the robot's movements to match actions in best policy"""
        if (self.q_policy != None):
            while (self.is_close((self.curr_x, self.curr_y)) == False):
                # retreive best action at current position
                action = self.q_policy[(self.curr_x, self.curr_y)]
                print(action)

                # move upwards
                if action == 'up': 
                    self.rotate_abs(math.pi/2)
                    self.translate(1)
                    self.curr_y = self.curr_y + 1
                
                # move downwards
                elif action == 'down':
                    self.rotate_abs(-math.pi/2)
                    self.translate(1)
                    self.curr_y = self.curr_y - 1

                # moving to the right
                elif action == 'right':
                    self.rotate_abs(0)
                    self.translate(1)
                    self.curr_x = self.curr_x + 1

                # moving to the left
                elif action == 'left':
                    self.rotate_abs(-math.pi)
                    self.translate(1)
                    self.curr_x = self.curr_x - 1
        return True
    
    def translate(self, distance):
        """Moves the robot in a straight line"""
        duration = distance/VELOCITY
        self.move(VELOCITY, 0, rospy.get_rostime(), duration)
    
    def rotate_abs(self, target_angle):
        """
        Turns the robot so that the new pose of the robot
        matches the target angle
        """
        sign = "" # determines the direction of rotation
        angular_distance = abs(target_angle - self.curr_angle)
        
        if target_angle > self.curr_angle:
            # rotate in a counter-clockwise direction
            sign = "positive"
        else:
            # rotate in a clockwise direction
            sign = "negative"

        duration = angular_distance/ANG_VELOCITY

        if sign == "positive":
            self.move(0, ANG_VELOCITY, rospy.get_rostime(), duration)
        else:
            self.move(0, -ANG_VELOCITY, rospy.get_rostime(), duration)
        
        self.curr_angle = target_angle # update the current position of the robot


    def is_close(self, curr_loc):
        """checks if distance between current loc and target below threshold"""
        distance = math.sqrt((curr_loc[0] - self.target_loc[0])**2 + (curr_loc[1] - self.target_loc[1])**2)

        if distance < 0.2:
            return True
        
        return False


if __name__ == "__main__":
    rospy.init_node("movement")
    p = qMove()
    rospy.sleep(2)
    rospy.spin()
