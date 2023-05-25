# Utility module for handling transformations
# Author: Kevin Zhou

import tf # pylint: disable=import-error
import rospy # module for ROS APIs pylint: disable=import-error
import numpy as np

class Transform:
    """Utility class for handling transformations"""
    def __init__(self):
        self.transform_listener = tf.TransformListener()
    def get_transformation_matrix(self, target, source, time=rospy.Time(0)):
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
    def transform_2d_point(self, transformation_matrix, point):
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
