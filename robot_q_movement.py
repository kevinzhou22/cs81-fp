#!/usr/bin/env python
# Author: Amit Das

# python/class libraries
import numpy as np
import q_learning
import math
import random
from transform import Transform

# rospy libraries
import rospy
import tf
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

# CONSTANTS
VELOCITY = 0.5 #m/s
ANG_VELOCITY = math.pi/4.0 #rad/s
DEFAULT_SCAN_TOPIC = 'robot_0/base_scan'
DEFAULT_OBJ_TOPIC = 'stalked'
MIN_SCAN_ANGLE_RAD = -40.0 / 180 * math.pi #rad
MAX_SCAN_ANGLE_RAD = +40.0 / 180 * math.pi #rad
MIN_THRESHOLD_DISTANCE = 0.5 #m

class Grid:
    """Initializes a Grid class which stores height, width, resolution data."""

    def __init__(self, occupancy_grid_data, width, height, resolution):
        self.grid = np.reshape(occupancy_grid_data, (height, width))
        self.resolution = resolution

class qMove:
    """
    Subscribes to get the point of the object in motion. Handles the 
    movement of the robot following policies returned by Q-learning.
    """

    def __init__(self, scan_angle=[MIN_SCAN_ANGLE_RAD, MAX_SCAN_ANGLE_RAD], min_threshold_distance=MIN_THRESHOLD_DISTANCE):
        """Initialization."""

        # Setting up subscribers for receiving data about laser, map, and moving object
        self._laser_sub = rospy.Subscriber(DEFAULT_SCAN_TOPIC, LaserScan, self._laser_callback, queue_size=1)
        self.sub = rospy.Subscriber("map", OccupancyGrid, self.map_callback, queue_size=1)
        self._obj_sub = rospy.Subscriber(DEFAULT_OBJ_TOPIC, PoseStamped, self._obj_callback, queue_size=1) 

        # setting up a publisher to publish velocity messages
        self.publisher = rospy.Publisher("/robot_0/cmd_vel", Twist, queue_size=1)
        self.vel_msg = Twist()
        self.rate = rospy.Rate(10)

        # instance variables
        self.q_policy = None # table to hold policies at different states
        self.map = None # the variable containing the map.
        self.height = None # height of the map
        self.width = None # width of the map
        self.curr_x = None # current x position
        self.curr_y = None # current y position
        self.curr_angle = None # current angle
        self.target_loc = None # location of moving object
        self.map_T_odom = np.array([[1, 0, 0, 5],
                                    [0, 1, 0, 5],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        self.transform = Transform() # access to transformations
        
        # Flag variables
        self.done_travelling = False # has robot taken steps to moving object
        self._close_obstacle = False # is robot close to a wall or object

        # information of laser scan/object detection
        self.scan_angle = scan_angle
        self.min_threshold_distance = min_threshold_distance
        self.close_to_robot = False

    
    def _obj_callback(self, msg):
        """
        Called upon receipt of point which represents the 
        belief about the current position of the moving object
        """

        self.done_travelling = False 
        
        # Find the position of the robot in odom reference frame
        odom_T_bl = self.transform.get_transformation_matrix('robot_0/odom', 'robot_0/base_link')
        yaw = self.transform._get_rotation('robot_0/base_link', 'robot_0/odom')
        robot_loc_odom = self.transform.transform_2d_point(odom_T_bl, ((0, 0)))

        # Find the position of the robot in map reference frame
        robot_loc_map = self.transform.transform_2d_point(self.map_T_odom, robot_loc_odom)
        self.curr_x = int(round(robot_loc_map[0]))
        self.curr_y = int(round(robot_loc_map[1]))
        self.curr_angle = yaw
        
        # position of stalker in map
        stalker_map = self.transform.transform_2d_point(self.map_T_odom, ((msg.pose.position.x, msg.pose.position.y)))
        self.target_loc = (int(round(stalker_map[0])), int(round(stalker_map[1])))

        # if too close to obstacle, robot rotates
        if self._close_obstacle == True:
            print("Detected obstacle! Rotating randomly")
            rand_angle = random.uniform(-math.pi, math.pi)
            start_time = rospy.get_rostime()
            total_time = abs(rand_angle / ANG_VELOCITY)
            
            # rotating robot at angular velocity for calculated amount fo time
            if (rand_angle > 0):
                self.move(0, ANG_VELOCITY, start_time, total_time)
            else:
                self.move(0, -ANG_VELOCITY, start_time, total_time)
            
            self._close_obstacle = False

        while (self.done_travelling != True):
            print("Targeted Location Locked!")
            print(self.target_loc)
            # Building table to target location as part of Q-learning
            self.get_table()
            # following policy generated by table
            self.done_travelling = self.follow_policy()

    def _laser_callback(self, msg):
        """Called upon receipt of message from laser"""

        # Determines whether robot is close to an obstacle
        if not self._close_obstacle:
            min_index = max(int(np.floor((self.scan_angle[0] - msg.angle_min) / msg.angle_increment)), 0)
            max_index = min(int(np.ceil((self.scan_angle[1] - msg.angle_min) / msg.angle_increment)), len(msg.ranges)-1)

            # finds the minimum range value in ranges between min_index and max_index
            min_range_val = float('inf')
            for i in range(len(msg.ranges)):
                if i > min_index and i < max_index:
                    if msg.ranges[i] < min_range_val:
                        min_range_val = msg.ranges[i]

            # determines whether obstacle is close enough to robot
            if (min_range_val < self.min_threshold_distance):
                self._close_obstacle = True
    
    def map_callback(self, msg):
        """Initializes map as Grid object; calls method to get best policy"""

        # create grid object upon receipt of map
        self.map = Grid(msg.data, msg.info.width, msg.info.height, msg.info.resolution)
        self.height  = int(msg.info.height*msg.info.resolution)
        self.width = int(msg.info.width*msg.info.resolution)

        # if policy has not been created; generate table to create policy
        if self.q_policy == None:
            if self.target_loc != None:
                self.get_table()

    def get_table(self):
        """Handles Q-learning; sets q_policy instance variable to best policy found after training"""
        # check to see if map has been read
        if self.map != None:
            # start the q-learning
            q_model = q_learning.QLearning(width=self.width, height=self.height, start_loc=(self.curr_x, self.curr_y), target_loc=self.target_loc)
            q_model.is_close_to_robot = self.close_to_robot
            q_model.training(10) # training for 10 iterations
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
        total_steps = 0
        if (self.q_policy != None):
            while (self.is_close((self.curr_x, self.curr_y)) == False and total_steps < 2):
                
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
                
                total_steps += 1
        
        return True
    
    def translate(self, distance):
        """Moves the robot in a straight line for a given distance"""
        duration = distance/VELOCITY
        self.move(VELOCITY, 0, rospy.get_rostime(), duration)
    
    def rotate_abs(self, target_angle):
        """
        Turns the robot so that the new pose of the robot
        matches the target angle
        """
        sign = "" # to determine the direction of rotation
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
        
        self.curr_angle = target_angle # update the current orientation of the robot


    def is_close(self, curr_loc):
        """checks if distance between current and target loc below threshold"""
        distance = math.sqrt((curr_loc[0] - self.target_loc[0])**2 + (curr_loc[1] - self.target_loc[1])**2)

        if distance < 1.1:
            self.close_to_robot = True
            return True
        
        return False

if __name__ == "__main__":
    rospy.init_node("movement")
    p = qMove()
    rospy.sleep(2)
    rospy.spin()
