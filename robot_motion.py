#!/usr/bin/env python

"""
Author: Amel Docena


"""
import time
import rospy
import actionlib
from geometry_msgs.msg import Twist, Pose, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from follower_dqn.msg import make_actionAction, make_actionFeedback, make_actionResult
import tf
import math

TOPICS = {'ODOM': '/robot_0/odom',
          'CMD_VEL': '/robot_0/cmd_vel',
          'SCAN': '/robot_0/base_scan'}

# Frequency at which the loop operates
LASER_FREQUENCY = 1 #Laser frequency

# Threshold of minimum clearance distance
MIN_THRESHOLD_DISTANCE = 0.90  # m, front obstacle threshold distance

# Field of view in radians that is checked in front of the robot
MIN_FRONT_SCAN_ANGLE_RAD = (-5.0 / 180) * math.pi;
MAX_FRONT_SCAN_ANGLE_RAD = (+5.0 / 180) * math.pi;

# To the left of the robot
eps = 35
MIN_LEFT_SCAN_ANGLE_RAD = (90.0-eps)*(math.pi/180)
MAX_LEFT_SCAN_ANGLE_RAD = (90.0+eps)*(math.pi/180)

# Right of the robot
MIN_RIGHT_SCAN_ANGLE_RAD = -((90.0+eps)*(math.pi/180))
MAX_RIGHT_SCAN_ANGLE_RAD = -((90.0-eps)*(math.pi/180))

# Scan angles: front, left, right
SCAN_ANGLES = {'f': (MIN_FRONT_SCAN_ANGLE_RAD, MAX_FRONT_SCAN_ANGLE_RAD),
               'l': (MIN_LEFT_SCAN_ANGLE_RAD, MAX_LEFT_SCAN_ANGLE_RAD),
               'r': (MIN_RIGHT_SCAN_ANGLE_RAD, MAX_RIGHT_SCAN_ANGLE_RAD)}

DIST_THRESH = {'min_front_dist': 0.50,
               'max_wall_thresh': 2,}

class Robot:
    def __init__(self, node_name, topics=TOPICS, scan_angles=SCAN_ANGLES, min_thresh_distance=MIN_THRESHOLD_DISTANCE, laser_freq=LASER_FREQUENCY):
        rospy.init_node(node_name)
        self.min_threshold_distance = min_thresh_distance
        self.scan_angles = scan_angles

        #Publishers/subscribers
        #Subscribe to robot odom
        rospy.Subscriber(topics['ODOM'], Odometry, self._set_robot_pose_cb, queue_size=1)
        rospy.Subscriber(topics['SCAN'], LaserScan, self._laser_cb, queue_size=laser_freq)
        rospy.Subscriber('/stalked', PoseStamped, self._set_object_pose_cb, queue_size=1)
        self._robot_cmd_pub = rospy.Publisher(topics['CMD_VEL'], Twist, queue_size=1)

        #Publish d_front, d_left, d_right
        self._d_front_pub = rospy.Publisher('/robot_0/d_front', Float32, queue_size=1)
        self._d_left_pub = rospy.Publisher('/robot_0/d_left', Float32, queue_size=1)
        self._d_right_pub = rospy.Publisher('/robot_0/d_right', Float32, queue_size=1)

        #Action server to move robot
        self._make_action_server = actionlib.SimpleActionServer("/make_action_server", make_actionAction,
                                                                execute_cb=self._make_action_cb, auto_start=False)
        self._make_action_server.start()

        #Parameters and variables
        self.move_distance = 1 #distance to move forward
        self.linear_velocity = 1
        self.angular_velocity = math.pi/4
        self.action_space = ['F', 'B', 'L', 'R', 'FL', 'FR', 'BL', 'BR'] #Action space
        self.curr_robot_pose = None
        self.curr_object_pose = None
        self.d_front, self.d_left, self.d_right = None, None, None

    #Callback methods
    def _set_robot_pose_cb(self, msg):
        #Sets curr robot pose
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.curr_robot_pose = (x, y, theta)

    def _set_object_pose_cb(self, msg):
        #Sets followed object pose
        x, y = msg.pose.position.x, msg.pose.position.y
        orientation = msg.pose.orientation
        _, _, theta = tf.transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.curr_object_pose = (x, y, theta)

    def _laser_cb(self, msg):
        """Processing of laser message.
        # Gather info for Front, Left, Right obstacle
        # Access to the index of the measurement in front, left, and right of the robot.
        # NOTE: assumption: the one at angle 0 corresponds to the front.
        """
        scan_readings = msg.ranges #laser scan readings

        #Front obstacle
        front_min_idx, front_max_idx = self.scan_range_indices(msg.angle_min, msg.angle_increment, self.scan_angles['f'])
        self.d_front = min(scan_readings[front_min_idx:front_max_idx])
        # if self.d_front <= self.min_threshold_distance:
        #     self.stop()

        #Left obstacle
        left_min_idx, left_max_idx = self.scan_range_indices(msg.angle_min, msg.angle_increment, self.scan_angles['l'])
        self.d_left = min(scan_readings[left_min_idx:left_max_idx])

        #Right obstacle
        right_min_idx, right_max_idx = self.scan_range_indices(msg.angle_min, msg.angle_increment, self.scan_angles['r'])
        self.d_right = min(scan_readings[right_min_idx:right_max_idx])

    def scan_range_indices(self, msg_angle_min, msg_angle_inc, angle_range):
        """
        Returns the indices of the range of laser scan angles
        :return:
        """
        min_angle, max_angle = angle_range
        min_idx = int((min_angle - msg_angle_min) / msg_angle_inc)
        max_idx = min_idx + int(abs(max_angle - min_angle) / msg_angle_inc) #abs to handle negative angle

        return min_idx, max_idx

    def get_robot_pose(self):
        """
        Returns robot pose
        Returns:

        """
        return self.curr_robot_pose

    def get_object_pose(self):
        """
        Returns object pose
        Returns:

        """
        return self.curr_object_pose

    def _make_action_cb(self, msg):
        """
        Makes action by the msg of velocities, linear and angular
        Args:
            msg:

        Returns:

        """
        feedback = make_actionFeedback()
        result = make_actionResult()

        while self.curr_robot_pose is None or self.curr_object_pose is None:
            time.sleep(1)

        action_idx = msg.action_idx
        success = self.take_action(action_idx, feedback)
        if success:
            states = [self.curr_robot_pose[0], self.curr_robot_pose[1], self.curr_robot_pose[2],
                      self.curr_object_pose[0], self.curr_object_pose[1], self.curr_object_pose[2],
                      self.d_front, self.d_left, self.d_right]
            result.states = states
            self._make_action_server.set_succeeded(result)

    #Motion methods
    def stop(self):
        """Stop the robot."""
        twist_msg = Twist()
        self._robot_cmd_pub.publish(twist_msg)

    def move(self, linear_vel, angular_vel, duration):
        """Send a velocity command (linear vel in m/s, angular vel in rad/s) for a given duration."""
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        start_time = rospy.get_rostime()

        while rospy.get_rostime() - start_time <= duration:
            self._robot_cmd_pub.publish(twist_msg)

    def compute_bearing(self, theta_goal):
        """
        Returns angular velocity and duration to correct the bearing from self.theta adding goal_theta.
        Note: reference frame is odom
        :param angle:
        :return:
        """
        correction = theta_goal #- self.theta
        w = self.angular_velocity
        if correction / w < 0:
            w *= -1
        duration = rospy.Duration(correction / w)
        return w, duration

    def compute_distance_from_self(self, pose):
        """
        Computes distance of pose from self
        Note: odom ref frame
        :param pose: tuple, x,y pose
        :return:
        """
        distance = math.sqrt((self.x - pose[0]) ** 2 + (self.y - pose[1]) ** 2)
        return distance

    def correct_bearing(self, w, duration):
        """
        Corrects the bearing of the robot, rotating in place by angular velocity w for set duration
        Note: odom ref frame
        :param w:
        :param duration:
        :return:
        """
        self.move(0, w, duration)

    def rotate_in_place(self, theta):
        """
        Rotates in place by theta
        Note: odom ref frame
        :return:
        """
        w, duration = self.compute_bearing(theta)
        self.correct_bearing(w, duration)

    def move_forward(self, dist):
        """
        Moves the robot forward for given distance
        Note: odom ref frame
        :return:
        """
        assert (self.linear_velocity > 0)
        duration = rospy.Duration(dist / self.linear_velocity)
        self.move(self.linear_velocity, 0, duration)

    def take_action(self, action_idx, feedback=None):
        """
        Moves robot by taking specific action
        Action space: ['F', 'B', 'L', 'R', 'FL', 'FR', 'BL', 'BR']
        Args:
            action_idx: index to action space

        Returns:

        """
        if feedback is not None:
            feedback.action_status = 'Robot taking action: ' + str(action_idx)
            self._make_action_server.publish_feedback(feedback)

        action = self.action_space[action_idx]
        assert action in self.action_space, "Invalid action--not in action space!"

        if action == 'F':
            angle = 0
        elif action == 'B':
            angle = math.pi
        elif action == 'L':
            angle = math.pi/2
        elif action == 'R':
            angle = -math.pi/2
        elif action == 'FL':
            angle = math.pi/4
        elif action == 'FR':
            angle = -math.pi/4
        elif action == 'BL':
            angle = 3*math.pi/4
        elif action == 'BR':
            angle = -3*math.pi/4

        self.rotate_in_place(angle)
        self.stop()
        self.move_forward(self.move_distance)
        self.stop()

        return True

    def spin(self):
        """
        Indefinte loop. We publish perceived front, left and right obstacles
        Returns:

        """
        while self.d_front is None or self.d_left is None or self.d_right is None:
            rospy.sleep(1)

        while not rospy.is_shutdown():
            self._d_front_pub.publish(self.d_front)
            self._d_left_pub.publish(self.d_left)
            self._d_right_pub.publish(self.d_right)
            rospy.sleep(1)

if __name__ == '__main__':
    robot = Robot('robot_motion')
    robot.spin()