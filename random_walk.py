#!/usr/bin/env python
#The line above is important so that this file is interpreted with Python when running it.

# Author: TODO: complete
# Date: TODO: complete

# Import of python modules.
import math # use of pi.
import random

# import of relevant libraries.
import rospy # module for ROS APIs pylint: disable=import-error
from geometry_msgs.msg import Twist # message type for cmd_vel pylint: disable=import-error
from sensor_msgs.msg import LaserScan # message type for scan pylint: disable=import-error

# Constants.
# Topic names
DEFAULT_CMD_VEL_TOPIC = 'robot_1/cmd_vel'
DEFAULT_SCAN_TOPIC = 'robot_1/base_scan' # name of topic for Stage simulator. For Gazebo, 'scan'

# Frequency at which the loop operates
FREQUENCY = 10 #Hz.

# Velocities that will be used (feel free to tune)
LINEAR_VELOCITY = 0.5 # m/s
ANGULAR_VELOCITY = math.pi/4 # rad/s

# Threshold of minimum clearance distance (feel free to tune)
MIN_THRESHOLD_DISTANCE = 0.75 # m, threshold distance, should be smaller than range_max

# Field of view in radians that is checked in front of the robot (feel free to tune)
MIN_SCAN_ANGLE_RAD = -45.0 / 180 * math.pi
MAX_SCAN_ANGLE_RAD = +45.0 / 180 * math.pi


class RandomWalk():
    def __init__(self, linear_velocity=LINEAR_VELOCITY, angular_velocity=ANGULAR_VELOCITY, min_threshold_distance=MIN_THRESHOLD_DISTANCE,
        scan_angle=[MIN_SCAN_ANGLE_RAD, MAX_SCAN_ANGLE_RAD]):
        """Constructor."""

        # Setting up publishers/subscribers.
        # Setting up the publisher to send velocity commands.
        self._cmd_pub = rospy.Publisher(DEFAULT_CMD_VEL_TOPIC, Twist, queue_size=1)
        # Setting up subscriber receiving messages from the laser.
        self._laser_sub = rospy.Subscriber(DEFAULT_SCAN_TOPIC, LaserScan, self._laser_callback, queue_size=1)

        # Parameters.
        self.linear_velocity = linear_velocity # Constant linear velocity set.
        self.angular_velocity = angular_velocity # Constant angular velocity set.
        self.min_threshold_distance = min_threshold_distance
        self.scan_angle = scan_angle
        
        # Flag used to control the behavior of the robot.
        self._close_obstacle = False # Flag variable that is true if there is a close obstacle.
        self._obstacle_encounter_time = 0 # Details time of encounter for encountered obstacle
        self._turn_time = 0 # total time to be turning
        self._encounter_angle = 0 # angle at which the obstacle was encountered

    def move(self, linear_vel, angular_vel):
        """Send a velocity command (linear vel in m/s, angular vel in rad/s)."""
        # Setting velocities.
        twist_msg = Twist()

        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self._cmd_pub.publish(twist_msg)
    def move_for_duration(self, linear_vel, angular_vel, duration):
        """
        Move for the specified velocity for a duration (linear vel in m/s,
        angular vel in rad/s, duration in s)
        """
        rate = rospy.Rate(FREQUENCY)
        start_time = rospy.get_rostime()
        end_time = start_time + rospy.Duration(duration)
        while not rospy.is_shutdown() and rospy.get_rostime() < end_time:
            self.move(linear_vel, angular_vel)
            rate.sleep()

    def stop(self):
        """Stop the robot."""
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)

    def _laser_callback(self, msg):
        """Processing of laser message."""
        # NOTE: assumption: the one at angle 0 corresponds to the front.

        if not self._close_obstacle:
            # Find the minimum range value between min_scan_angle and
            # max_scan_angle

            # If the minimum range value is closer to min_threshold_distance, change the flag self._close_obstacle
            # Note: You have to find the min index and max index.
            # Please double check the LaserScan message http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html
            ####### ANSWER CODE BEGIN #######
            if self._close_obstacle:
                # already turning to deal with the obstacle
                return
            # find releant indices for distances that are in desired field of view
            min_index = int(max(math.ceil((self.scan_angle[0] - msg.angle_min) / msg.angle_increment), 0))
            max_index = int(min(math.floor((self.scan_angle[1] - msg.angle_min) / msg.angle_increment), len(msg.ranges) - 1))
            min_dist = float('inf')
            min_dist_angle = None
            # analyze message to find minimum distance within field of view
            for i in range(min_index, max_index + 1):
                dist = msg.ranges[i]
                angle = msg.angle_min + i * msg.angle_increment
                if dist < msg.range_min or dist > msg.range_max:
                    # invalid angle
                    continue
                if dist < min_dist:
                    min_dist = dist
                    min_dist_angle = angle

            if min_dist < self.min_threshold_distance:
                # we want to turn for roughly a given angle, and to do so, we need to leverage knowledge
                # of time spent turning and target time spent turning
                self._close_obstacle = True
                self._obstacle_encounter_time = rospy.get_rostime()
                self._encounter_angle = min_dist_angle
                turn_amount = random.uniform(0, math.pi)
                self._turn_time = turn_amount / self.angular_velocity
            
            
            ####### ANSWER CODE END #######

    def spin(self):
        rate = rospy.Rate(FREQUENCY) # loop at 10 Hz.

        while not rospy.is_shutdown():
            # Keep looping until user presses Ctrl+C
            
            # If the flag self._close_obstacle is False, the robot should move forward.
            # Otherwise, the robot should rotate for a random amount of time
            # after which the flag is set again to False.
            # Use the function move already implemented, passing the default velocities saved in the corresponding class members.

            ####### ANSWER CODE BEGIN #######
            if self._close_obstacle and self._obstacle_encounter_time + rospy.Duration(self._turn_time) < rospy.get_rostime():
                self._close_obstacle = False
                # return instead of continuing as the obstacle may still be in the way
                # sleep half a second to give time to pick up additional potential obstacle readings
                sleep_rate = rospy.Rate(2)
                sleep_rate.sleep()
                continue

            if self._close_obstacle:
                rotation_msg = Twist()
                # choose direction to turn based on where object was detected
                if self._encounter_angle <= 0:
                    rotation_msg.angular.z = -self.angular_velocity
                else:
                    rotation_msg.angular.z = self.angular_velocity
                self.move_for_duration(0, rotation_msg.angular.z, self._turn_time)
            else:
                # default: move forward
                forward_msg = Twist()
                forward_msg.linear.x = LINEAR_VELOCITY
                self._cmd_pub.publish(forward_msg)
            
            ####### ANSWER CODE END #######

            rate.sleep()
        

def main():
    """Main function."""

    # 1st. initialization of node.
    rospy.init_node("random_walk")

    # Sleep for a few seconds to wait for the registration.
    rospy.sleep(2)

    # Initialization of the class for the random walk.
    random_walk = RandomWalk()

    # If interrupted, send a stop command before interrupting.
    rospy.on_shutdown(random_walk.stop)

    # Robot random walks.
    try:
        random_walk.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("ROS node interrupted.")


if __name__ == "__main__":
    """Run the main function."""
    main()
