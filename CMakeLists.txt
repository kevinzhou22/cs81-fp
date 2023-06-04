## For full information http://wiki.ros.org/catkin/CMakeLists.txt 

#############################################################################
## Set minimum required version of cmake, project name and compile options ##
#############################################################################
cmake_minimum_required(VERSION 3.0.2)
project(follower_dqn)

###########################################################################
## Find catkin packages and libraries for catkin and system dependencies ##
###########################################################################
find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  actionlib_msgs
  message_generation
  nav_msgs
)

##########################################
## Setup for python modules and scripts ##
##########################################
catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################
# add_service_files(
#    FILES
# )

# No new message declared.
add_action_files(
    FILES
        make_action.action
        )

generate_messages(
    DEPENDENCIES actionlib_msgs std_msgs nav_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################
# No new parameters declared.

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS
    rospy
    geometry_msgs
    std_msgs
    actionlib_msgs
    nav_msgs
  DEPENDS message_runtime
)

###########
## Build ##
###########
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

catkin_install_python(
    PROGRAMS
        dqn.py
        reset_simulation.py
        robot_motion.py
        random_walk.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############
# No unit test performed.
