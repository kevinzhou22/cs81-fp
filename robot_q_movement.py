#!/usr/bin/env python

# python/class libraries
import numpy as np
import q_learning
import math

# rospy libraries
import rospy
from geometry_msgs.msg import Twist # message type
from nav_msgs.msg import OccupancyGrid

# CONSTANTS
# Frequency at which the loop operates
VELOCITY = 0.2 #m/s
ANG_VELOCITY = math.pi/4.0 #rad/s
DEFAULT_SCAN_TOPIC = 'base_scan'

class Grid:
    """Initializes a Grid class which stores height, width, resolution data."""

    def __init__(self, occupancy_grid_data, width, height, resolution):
        self.grid = np.reshape(occupancy_grid_data, (height, width))
        self.resolution = resolution

class qMove:
    def __init__(self):
        """Initialization."""
        self.sub = rospy.Subscriber("map", OccupancyGrid, self.map_callback, queue_size=1)
        self.publisher = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.vel_msg = Twist()
        self.rate = rospy.Rate(5)
        self.q_policy = None # table holding policies at different states
        self.map = None # the variable containing the map.
        self.height = None # height of the map
        self.width = None # width of the map
        self.curr_x = 5 # holds the current x position
        self.curr_y = 5 # holds the current y position
        self.curr_angle = 0 # holds the current angle
        self.target_loc = (2, 2)
        

    def map_callback(self, msg):
        """Initializes map as Grid object; calls method to get best policy"""
        # create grid object upon receipt of map
        self.map = Grid(msg.data, msg.info.width, msg.info.height, msg.info.resolution)
        self.height  = int(msg.info.height*msg.info.resolution)
        self.width = int(msg.info.width*msg.info.resolution)

        print("Height and Width:")
        print(self.height)
        print(self.width)
        # if policy has not been created; generate table
        if self.q_policy == None:
            self.get_table()

    def get_table(self):
        """Handles Q-learning; sets q_policy instance variable to policy found after training"""
        # check to see if map has been read
        if self.map != None:
            # start the q-learning
            q_model = q_learning.QLearning(width=self.width, height=self.height, start_loc=(self.curr_x, self.curr_y), target_loc=self.target_loc)
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
        if (self.q_policy != None):
            while (self.is_close((self.curr_x, self.curr_y)) == False):
                # retreive best action at current position
                action = self.q_policy[(self.curr_x, self.curr_y)]

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
    
    def translate(self, distance):
        """Moves the robot in a straight line"""
        duration = distance/VELOCITY
        self.move(VELOCITY, 0, rospy.get_rostime(), duration)
    
    def rotate_abs(self, target_angle):
        """
        Turns the robot so that the new pose of the robot
        matches the target angle
        """
        duration = (target_angle - self.curr_angle)/ANG_VELOCITY

        if duration > 0:
            self.move(0, ANG_VELOCITY, rospy.get_rostime(), duration)
        else:
            self.move(0, -ANG_VELOCITY, rospy.get_rostime(), -duration)
        
        self.curr_angle = target_angle


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
    
    p.get_table()
    p.follow_policy()

    rospy.spin()
