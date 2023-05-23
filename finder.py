#!/usr/bin/env python
#The line above is important so that this file is interpreted with Python when running it.

# Author: Kevin Zhou
# Date: April 12, 2023

# Import of python modules.
import sys
import math

# import of relevant libraries.
import rospy # module for ROS APIs pylint: disable=import-error
from geometry_msgs.msg import Twist, Pose # message type for cmd_vel pylint: disable=import-error
from sensor_msgs.msg import LaserScan # message type for scan pylint: disable=import-error
from nav_msgs.msg import OccupancyGrid, MapMetaData # pylint: disable=import-error

import tf # pylint: disable=import-error
import numpy as np
import matplotlib.pyplot as plt
from lock_on import MovingObjectDetector

# Topic names
DEFAULT_SCAN_TOPIC = 'robot_0/base_scan' 
DEFAULT_OCCUPANCY_GRID_TOPIC = 'map'
DEFAULT_CMD_VEL_TOPIC = 'robot_0/cmd_vel'


# Frame names
DEFAULT_SCAN_FRAME_ID = 'robot_0/base_laser_link'
DEFAULT_ODOM_FRAME_ID = 'robot_0/odom'
# Parameters
DEFAULT_RESOLUTION = 0.1 # m
DEFAULT_GRID_WIDTH =  150 # grid cells
DEFAULT_GRID_HEIGHT = 150 # grid cells
DEFAULT_GRID_TRANSLATION = (-6, -6) # m
DEFAULT_GRID_ROTATION = 0 # yaw
LASER_MIN_ERROR = 0.05 # m 
LASER_PERCENT_INCREASE_ACCURACY = 0.002 # percent
INITIALIZATION_DURATION = 15 # s

# Frequency at which the loop operates
LOOP_AND_PUBLISH_FREQUENCY = 1 #Hz.
UPDATE_FREQUENCY = 0.5
DETECT_FREQUENCY = 2

TWO_PI = np.pi * 2

# Detector from lock_on.py
detector = MovingObjectDetector(eps=1.5, min_samples=4)

class Mode:
    BASELINE_INITIALIZATION = 0
    MOVEMENT_DETECTION = 1

class Finder():
    """Node for assignment"""
    def __init__(
        self,
        loop_and_publish_frequency=LOOP_AND_PUBLISH_FREQUENCY,
        update_frequency=UPDATE_FREQUENCY,
        detect_frequency=DETECT_FREQUENCY,
        scan_topic=DEFAULT_SCAN_TOPIC,
        grid_topic=DEFAULT_OCCUPANCY_GRID_TOPIC,
        cmd_vel_topic=DEFAULT_CMD_VEL_TOPIC,
        scan_frame_id=DEFAULT_SCAN_FRAME_ID,
        odom_frame_id=DEFAULT_ODOM_FRAME_ID,
        resolution=DEFAULT_RESOLUTION,
        grid_width=DEFAULT_GRID_WIDTH,
        grid_height=DEFAULT_GRID_HEIGHT,
        grid_translation=DEFAULT_GRID_TRANSLATION,
        grid_rotation=DEFAULT_GRID_ROTATION,
        laser_min_error=LASER_MIN_ERROR,
        laser_percent_increase_accuracy=LASER_PERCENT_INCREASE_ACCURACY,
        initialization_duration=INITIALIZATION_DURATION
        ):
        """Constructor."""

        # Setting up publishers/subscribers.
        self._cmd_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
        self._grid_pub = rospy.Publisher(grid_topic, OccupancyGrid, queue_size=1)
        self._anom_pub = rospy.Publisher('anom', OccupancyGrid, queue_size=1)
        self._laser_sub = rospy.Subscriber(scan_topic, LaserScan, self._laser_callback, queue_size=1)
        self.transform_listener = tf.TransformListener()

        # frames
        self.scan_frame_id=scan_frame_id
        self.odom_frame_id=odom_frame_id

        # parameters
        self.loop_and_publish_frequency = loop_and_publish_frequency
        self.update_interval = rospy.Duration(1 / update_frequency)
        self.detect_interval = rospy.Duration(1 / detect_frequency)
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.resolution = resolution
        self.laser_min_error = laser_min_error
        self.laser_percent_increase_accuracy = laser_percent_increase_accuracy
        self.initialization_duration = rospy.Duration(initialization_duration)
        # state
        # (free, occupied, unknown)
        self.grid = np.tile(np.array([0, 0, 1], dtype=np.float64),(grid_height, grid_width, 1),)
        self.anomalous_grid = np.zeros((grid_height, grid_width))
        self.anomalous = set()
        self.last_update = rospy.get_rostime()
        self.last_detect = rospy.get_rostime()
        self.mode = Mode.BASELINE_INITIALIZATION

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
        self.points = []

    
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
        x, y = point
        transformed_point = np.matmul(transformation_matrix, np.array([x, y, 0, 1], dtype='float64'))
        return (transformed_point[0], transformed_point[1])

    def _get_coordinates_from_angle_and_distance(self, angle, distance):
        return (np.cos(angle) * distance, np.sin(angle) * distance)
    def _get_laser_accuracy(self, laser_distance):
        return self.laser_min_error + laser_distance * self.laser_percent_increase_accuracy
    def _get_perpendicular_ray_coeff(self, horizontal_component, ray_distance, angle_increment):
        # we scale the std_dev with the sqrt of 1/2 the gap between rays
        # this is to reflect that as we go farther, the exact middle of 
        std_dev = abs(np.sin(angle_increment / 2)) * np.sqrt(ray_distance)
        coeff = np.exp(- (horizontal_component ** 2) / (2 * (std_dev ** 2)))
        return coeff

    def _get_parallel_ray_coeffs(self, laser_distance, parallel_component):
        x = parallel_component - laser_distance
        std_dev = self._get_laser_accuracy(laser_distance)
        # these equations manually tuned to resemble those of a 2017 research paper on the topic
        free = 1 / (1 + np.exp(30 * (x +  std_dev)))
        unknown = 1 / (1 + np.exp(-30 * (x - std_dev)))
        occupied = 1 - free - unknown
        return np.array([free, occupied])

    def _fuse_measurement_for_grid_point(self, grid_point, new):
        # Dempster's rule of combination as described in 
        # OCCUPANCY MODELLING FOR MOVING OBJECT DETECTION FROM LIDAR POINT CLOUDS: A COMPARATIVE STUDY
        # by W. Xiao et al
        old = self.grid[grid_point[1]][grid_point[0]]
        conflict = old[1] * new[0] + old[0] * new[1]
        conflict_coefficient = 1 / (1 - conflict)
        combined_measurement = conflict_coefficient * np.array([
            old[0] * new[0] + old[0] * new[2] + old[2] * new[0],
            old[1] * new[1] + old[1] * new[2] + old[2] * new[1],
            old[2] * new[2]
        ], dtype=np.float64)
        combined_measurement = combined_measurement / np.linalg.norm(combined_measurement)
        self.grid[grid_point[1]][grid_point[0]] = combined_measurement
    def _update_grid_for_point(self, sorted_observations, grid_to_laser, grid_point, angle_increment):
        laser_point = self._transform_2d_point(grid_to_laser, grid_point)
        x, y = laser_point
        angle = (np.arctan2(y, x) + TWO_PI) % TWO_PI
        distance = np.linalg.norm(np.array(laser_point))
        # finding closest angle methodology taken from https://stackoverflow.com/a/26026189
        angles = sorted_observations[:, 0]
        idx = np.searchsorted(angles, angle, side="left")
        if idx > 0 and (idx == len(angles) or math.fabs(angle - angles[idx-1]) < math.fabs(angle - angles[idx])):
            idx = idx - 1
        closest_ray = sorted_observations[idx]
        angle_to_ray = angle - closest_ray[0]
        perp_component_to_ray = abs(np.sin(angle_to_ray)) * distance
        parallel_component_to_ray = abs(np.cos(angle_to_ray)) * distance
        # heuristic to avoid doing unnecessary calculations 
        if parallel_component_to_ray > closest_ray[1] + 5 * self._get_laser_accuracy(closest_ray[1]):
            parallel_coeffs = np.array([0, 0])
        elif parallel_component_to_ray < closest_ray[1] - 5 * self._get_laser_accuracy(closest_ray[1]):
            parallel_coeffs = np.array([1, 0])
        else:
            parallel_coeffs = self._get_parallel_ray_coeffs(closest_ray[1], parallel_component_to_ray)
        perp_coeff = self._get_perpendicular_ray_coeff(perp_component_to_ray, closest_ray[1], angle_increment)
        free, occupied = parallel_coeffs * perp_coeff
        new_measurement = np.array([free, occupied, 1 - free - occupied])
        self._fuse_measurement_for_grid_point(grid_point, new_measurement)

    def _update_grid(self, laser_to_odom, odom_to_laser, observations, max_range, angle_increment):
        """Observations are ((laser x, laser y), obstacle)"""
        odom_start = self._transform_2d_point(laser_to_odom, (0, 0))
        grid_start = self._transform_2d_point(self.odom_to_grid, odom_start)
        max_observed_range = 0
        for observation in observations:
            if observation[1] == max_range:
                continue
            max_observed_range = max(max_observed_range, observation[1])
        max_observed_range = min(max_observed_range / self.resolution + 1, max_range / self.resolution)
        grid_to_laser = np.matmul(odom_to_laser ,self.grid_to_odom)
        # (angle, distance per laser, was_obstacle_found)
        sorted_observations = observations[observations[:, 0].argsort()]
        # add "first" angle to end and "last" angle to front to reflect modular equivalence of angles
        last_obs = np.copy(sorted_observations[-1])
        last_obs[0] -= TWO_PI
        first_obs = np.copy(sorted_observations[0])
        first_obs[0] += TWO_PI
        sorted_observations = np.insert(sorted_observations, 0, last_obs, axis=0)
        sorted_observations = np.append(sorted_observations, np.array([first_obs]), axis=0)
        box_x_start = int(round(max(0, grid_start[0] - max_observed_range)))
        box_x_end = int(round(min(self.grid.shape[1] - 1, grid_start[0] + max_observed_range)))
        box_y_start = int(round(max(0, grid_start[1] - max_observed_range)))
        box_y_end = int(round(min(self.grid.shape[0] - 1, grid_start[1] + max_observed_range)))
        for y in range(box_y_start, box_y_end + 1):
            for x in range(box_x_start, box_x_end + 1):
                self._update_grid_for_point(sorted_observations, grid_to_laser, (x, y), angle_increment)

    def _extract_occupied_anomalies(self, laser_to_odom, observations):
            self.anomalous_grid = np.zeros((self.grid_height, self.grid_width))
            anomalous_odom_points = []
            for observation in observations:
                angle, distance, was_obstacle_found = observation
                if not was_obstacle_found:
                    continue
                laser_coords = self._get_coordinates_from_angle_and_distance(angle, distance)
                odom_coords = self._transform_2d_point(laser_to_odom, laser_coords)
                grid_coords = np.array(self._transform_2d_point(self.odom_to_grid, odom_coords))
                upper = np.ceil(grid_coords).astype(int)
                lower = np.floor(grid_coords).astype(int)
                upper_left = self.grid[upper[1], lower[0]]
                upper_right = self.grid[upper[1], upper[0]]
                lower_left = self.grid[lower[1], lower[0]]
                lower_right = self.grid[lower[1], upper[0]]
                x_diff = upper[0] - lower[0]
                y_diff = upper[1] - lower[1]
                left_coeff = (upper[0] - grid_coords[0]) / x_diff
                right_coeff = (grid_coords[0] - lower[0]) / x_diff 
                x_2 = left_coeff * upper_left + right_coeff * upper_right
                x_1 = left_coeff * lower_left + right_coeff * lower_right
                state = (upper[1] - grid_coords[1]) / y_diff * x_1 + (grid_coords[1] - lower[1]) / y_diff * x_2
                conflict = state[0]
                if conflict > 0.99:
                    anomalous_odom_points.append(odom_coords)
            return anomalous_odom_points
                
    def _laser_callback(self, msg):
        """Processing of laser message."""
        time = msg.header.stamp
        laser_to_odom = self._get_transformation_matrix(self.odom_frame_id, self.scan_frame_id, time)
        odom_to_laser = np.linalg.inv(laser_to_odom)
        time = rospy.get_rostime()
        should_update = time - self.last_update >= self.update_interval
        should_detect = time - self.last_detect >= self.detect_interval
        if not should_update and not should_detect:
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
        if self.mode == Mode.MOVEMENT_DETECTION and should_detect:
            anomalies = self._extract_occupied_anomalies(laser_to_odom, observations)
            print(anomalies)
            centroid1 = detector.detect_object(anomalies)
            # should print the (x, y) coordinates of the center of the object
            print("Object location:", centroid1, "\n")
            self.last_detect = time
        if should_update:
            self._update_grid(laser_to_odom, odom_to_laser, observations, msg.range_max, msg.angle_increment)
            self.last_update = time
    def move(self, linear_vel, angular_vel):
        """Send a velocity command (linear vel in m/s, angular vel in rad/s)."""
        # Setting velocities.
        twist_msg = Twist()

        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self._cmd_pub.publish(twist_msg)
    def stop(self):
        """Stop the robot."""
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)

    def publish(self):
        """Start publishing the occupancy grid"""
        rate = rospy.Rate(self.loop_and_publish_frequency)
        start_time = rospy.get_rostime()
        while not rospy.is_shutdown():
            if self.mode == Mode.BASELINE_INITIALIZATION:
                self.move(0, 0.3)
                if rospy.get_rostime() - start_time > self.initialization_duration:
                    self.stop()
                    self.mode = Mode.MOVEMENT_DETECTION                
            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = rospy.get_rostime()
            grid_msg.header.frame_id = self.odom_frame_id
            grid_msg.info = self.map_metadata
            grid_msg.data = np.rint(self.grid[:,:, 1] * 100).astype(int).flatten()
            self._grid_pub.publish(grid_msg)
            # grid_msg.data = self.anomalous_grid.flatten()
            # self._anom_pub.publish(grid_msg)
            rate.sleep()

def main():
    """Main function."""
    np.set_printoptions(threshold=sys.maxsize)
    # 1st. initialization of node.
    rospy.init_node("finder")

    # Sleep for a few seconds to wait for the registration.
    rospy.sleep(2)

    # Initialization of the class for the random walk.
    grid_mapper = Finder()

    # Robot moves.
    try:
        grid_mapper.publish()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")


if __name__ == "__main__":
    """Run the main function."""
    main()

