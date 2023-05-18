#!/usr/bin/env python
#The line above is important so that this file is interpreted with Python when running it.

# Author: Kevin Zhou
# Date: April 12, 2023

# Import of python modules.
import sys
import math

# import of relevant libraries.
import rospy # module for ROS APIs pylint: disable=import-error
from geometry_msgs.msg import Pose # message type for cmd_vel pylint: disable=import-error
from sensor_msgs.msg import LaserScan # message type for scan pylint: disable=import-error
from nav_msgs.msg import OccupancyGrid, MapMetaData # pylint: disable=import-error

import tf # pylint: disable=import-error
import numpy as np

# Topic names
DEFAULT_SCAN_TOPIC = 'base_scan' # scan if using turtlebot
DEFAULT_OCCUPANCY_GRID_TOPIC = 'map'

# Frame names
DEFAULT_SCAN_FRAME_ID = 'base_laser_link' # base_scan if using turtlebot

# Parameters
DEFAULT_RESOLUTION = 0.1 # m
DEFAULT_GRID_WIDTH = 800 # grid cells
DEFAULT_GRID_HEIGHT = 400 # grid cells
DEFAULT_GRID_TRANSLATION = (-10, -10) # m
DEFAULT_GRID_ROTATION = 0 # yaw

# Frequency at which the loop operates
PUBLISH_FREQUENCY = 5 #Hz.
UPDATE_FREQUENCY = 5

TWO_PI = np.pi * 2
class Finder():
    """Node for assignment"""
    def __init__(
        self,
        publish_frequency=PUBLISH_FREQUENCY,
        update_frequency=UPDATE_FREQUENCY,
        scan_topic=DEFAULT_SCAN_TOPIC,
        grid_topic=DEFAULT_OCCUPANCY_GRID_TOPIC,
        resolution=DEFAULT_RESOLUTION,
        grid_width=DEFAULT_GRID_WIDTH,
        grid_height=DEFAULT_GRID_HEIGHT,
        grid_translation=DEFAULT_GRID_TRANSLATION,
        grid_rotation=DEFAULT_GRID_ROTATION
        ):
        """Constructor."""

        # Setting up publishers/subscribers.
        self._grid_pub = rospy.Publisher(grid_topic, OccupancyGrid, queue_size=1)
        self._laser_sub = rospy.Subscriber(scan_topic, LaserScan, self._laser_callback, queue_size=1)
        self.transform_listener = tf.TransformListener()

        # parameters
        self.publish_frequency = publish_frequency
        self.update_frequency = update_frequency
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.resolution = resolution
        # state
        self.grid = np.full((grid_height, grid_width), -1)
        self.last_update = rospy.get_rostime()

        # for convenience, storing some grid variables that can be caluclated dynamically
        scale_ratio = resolution
        t = tf.transformations.translation_matrix((grid_translation[0], grid_translation[1], 0))
        R = tf.transformations.euler_matrix(0, 0, grid_rotation)
        scale = np.array([
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1]
        ])
        self.grid_to_odom = np.matmul(np.matmul(t, R), scale)
        self.odom_to_grid = np.linalg.inv(self.grid_to_odom)
        self.map_metadata = self._get_map_metadata(resolution, grid_width, grid_height, grid_translation, grid_rotation)

    
    def _get_map_metadata(self, resolution, width, height, translation, rotation):
        """Generates the metadata for the occupancy grid"""
        grid_origin = Pose()
        grid_origin.position.x = translation[0]
        grid_origin.position.y = translation[1  ]
        grid_origin.position.z = 0
        x, y, z, w = tf.transformations.quaternion_from_euler(0, 0, rotation)
        grid_origin.orientation.x = x
        grid_origin.orientation.y = y
        grid_origin.orientation.z = z
        grid_origin.orientation.w = w

        map_metadata = MapMetaData()
        map_metadata.map_load_time = rospy.get_rostime()
        map_metadata.resolution = resolution
        map_metadata.width = width
        map_metadata.height = height
        map_metadata.origin = grid_origin
        return map_metadata

    def _get_transformation_matrix(self, target, source):
        """Gets the transformation matrix from target to source, looping until found"""
        has_found_transformation = False
        # it's possible the relevant transformation has not been published yet
        # this just loops until the transformation is acquired
        while not has_found_transformation:
            try:
                (trans, rot) = self.transform_listener.lookupTransform(
                    target,
                    source, 
                    rospy.Time(0)
                )
                has_found_transformation = True
            except tf.LookupException:
                rospy.sleep(0.5)
                continue
        t = tf.transformations.translation_matrix(trans)
        R = tf.transformations.quaternion_matrix(rot)
        transformation_matrix = np.matmul(t, R)
        return transformation_matrix
    def _transform_2d_point(self, transformation_matrix, point):
        x, y = point
        transformed_point = np.matmul(transformation_matrix, np.array([x, y, 0, 1], dtype='float64'))
        return (transformed_point[0], transformed_point[1])

    def _get_coordinates_from_angle_and_distance(self, angle, distance):
        return (np.cos(angle) * distance, np.sin(angle) * distance)
    def _update_grid_for_point(self, sorted_observations, grid_to_laser, grid_point):
        laser_point = self._transform_2d_point(grid_to_laser, grid_point)
        x, y = laser_point
        angle = (np.arctan2(y, x) + TWO_PI) % TWO_PI
        distance = np.linalg.norm(np.array(laser_point))
        # taken from https://stackoverflow.com/a/26026189
        angles = sorted_observations[:, 0]
        idx = np.searchsorted(angles, angle, side="left")
        if idx > 0 and (idx == len(angles) or math.fabs(angle - angles[idx-1]) < math.fabs(angle - angles[idx])):
            idx = idx - 1
        closest_ray = sorted_observations[idx]
        print(closest_ray)
        print(distance)
        if distance > closest_ray[1]:
            return
        self.grid[int(grid_point[1]), int(grid_point[0])] = 0
        # elif abs(distance - closest_ray[1]) < 0.1 and closest_ray[2]:
        #     self.grid[int(grid_point[1]), int(grid_point[0])] = 100
        # else:
        #     self.grid[int(grid_point[1]), int(grid_point[0])] = 0


    def _update_grid(self, laser_to_odom, odom_to_laser, observations, max_range):
        """Observations are ((laser x, laser y), obstacle)"""
        odom_start = self._transform_2d_point(laser_to_odom, (0, 0))
        grid_start = self._transform_2d_point(self.odom_to_grid, odom_start)
        grid_cells_range = max_range / self.resolution
        grid_to_laser = np.matmul(odom_to_laser ,self.grid_to_odom)
        # (angle, distance per laser, was_obstacle_found)
        sorted_observations = observations[observations[:, 0].argsort()]
        last_obs = np.copy(sorted_observations[-1])
        last_obs[0] -= TWO_PI
        first_obs = np.copy(sorted_observations[0])
        first_obs[0] += TWO_PI
        sorted_observations = np.insert(sorted_observations, 0, last_obs, axis=0)
        sorted_observations = np.append(sorted_observations, np.array([first_obs]), axis=0)
        
        box_x_start = int(round(max(0, grid_start[0] - grid_cells_range)))
        box_x_end = int(round(min(self.grid.shape[0] - 1, grid_start[0] + grid_cells_range)))
        box_y_start = int(round(max(0, grid_start[1] - grid_cells_range)))
        box_y_end = int(round(min(self.grid.shape[1] - 1, grid_start[1] + grid_cells_range)))
        for y in range(box_y_start, box_y_end + 1):
            for x in range(box_x_start, box_x_end + 1):
                self._update_grid_for_point(sorted_observations, grid_to_laser, (x, y))

        # process all observations to convert angles into 0-360 (eventually: calculate std dev of models)
        # iterate through all grid cells in a bounding box
        # convert grid cell coordinates to laser and find angle in orientation
        
    def _laser_callback(self, msg):
        """Processing of laser message."""
        laser_to_odom = self._get_transformation_matrix('odom', 'base_laser_link')
        odom_to_laser = np.linalg.inv(laser_to_odom)
        time = rospy.get_rostime()
        if time - self.last_update < rospy.Duration(1 / self.update_frequency):
            return
        # (angle, distance, was_obstacle_found)
        observations = np.zeros((len(msg.ranges), 3))
        for i, dist in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if dist < msg.range_min or dist > msg.range_max:
                # bad reading
                continue
            is_end_occupied = True
            if dist == msg.range_max or dist == msg.range_min:
                # we assume msg.range_min means nothin was found, so
                # we make everything along that line free
                dist = msg.range_max
                # assume no object found
                is_end_occupied = False

            observations[i] = np.array([(angle + TWO_PI) % TWO_PI, dist, is_end_occupied])
        self._update_grid(laser_to_odom, odom_to_laser, observations, msg.range_max)
        self.last_update = time

    def publish(self):
        """Start publishing the occupancy grid"""
        rate = rospy.Rate(self.publish_frequency)
        self.grid = np.full((self.grid_height, self.grid_width), -1)
        while not rospy.is_shutdown():
            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = rospy.get_rostime()
            grid_msg.header.frame_id = 'odom'
            grid_msg.info = self.map_metadata
            grid_msg.data = self.grid.flatten()
            self._grid_pub.publish(grid_msg)
            rate.sleep()

def main():
    """Main function."""

    # 1st. initialization of node.
    rospy.init_node("finder")

    # Sleep for a few seconds to wait for the registration.
    rospy.sleep(2)

    # Initialization of the class for the random walk.
    grid_mapper = Finder()
    np.set_printoptions(threshold=sys.maxsize)

    # Robot moves.
    try:
        grid_mapper.publish()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")


if __name__ == "__main__":
    """Run the main function."""
    main()

