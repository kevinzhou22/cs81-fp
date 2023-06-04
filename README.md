# COSC 81: 23S
# Final Project: README
## Buddy Bot
### Description

The aim of this project is to create a robot that can learn to follow a nearby moving object.

It consists of two main components:
* A moving object detector
* A learner for following the moving object

## Code Files
* finder.py: maps the environment probabilistically and finds points that likely correspond to moving
objects
* lock-on.py: analyzes the output of `finder.py`, processing the data on detected moving objects. It calculates and updates the object's location, orientation, and linear velocity, publishing this information to a ROS topic for use by ROS nodes.
* transform.py: contains utility functions for conducting transformations between coordinate frames
and extracting information about relative translation and orientation.
* q_learning.py: implements q-learning. That is, includes functions to update q-table, calculate rewards,
start the training process and return the best policy as a result of training.
* robot_q_movement.py: processes a point published by `finder.py`; moves the robot to that point

For DQN:  
* dqn.py: DQN training and evaluation
* reset_simulation.py: resets the ROS world simulation
* robot_motion.py: robot node subscribed to by dqn.py
* package files: CMakeLists.txt, setup.py, package.xml, launch/simulation.launch


## Setup and Execution
(in addition to tf, get numpy and scikit-learn)
Setup:
```
sudo apt-get install python-sklearn python-sklearn-lib python-sklearn-doc
sudo apt-get install python-rosdep
sudo rosdep init
rosdep update
```

Then, in separate terminals, execute each of the commands:
```
roscore
```
```

rosrun stage_ros stageros PA1.world
```
```
python finder.py # motion detection
```
To move the other robot:
```
python random_walk.py # moving the followed robot
```
To run the main robot:
```
python robot_q_movement.py # executing the discrete learning algorithm for following
```

To train the dqn: Just run <code>python dqn.py</code> after running <code>roscore</code>. In the script, uncomment and supply parameters for <code>dqn.train_model(path, filename)</code>.
The trained model will be saved with extension <code>filename.h5</code>, as well as a record of sum of rewards per training episode stored in <code>filename.pkl</code>.

To evaluate the trained dqn: Run <code>python dqn.py</code>. In the script, uncomment and supply parameters for <code>dqn.eval_dqn(filename, n_iters)</code>

