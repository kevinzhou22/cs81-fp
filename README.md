# COSC 81: 23S
# Final Project: README
## Buddy Bot
### Description

The aim of this project is to create a robot that can learn to follow a nearby moving object.

It consists of two main components:
* A moving object detector
* A learner for following the moving object

## Code Files
* finder.py: maps the enviornment probabilistically and finds points that likely correspond to moving
objects
* lock-on.py: analyzes the output of `finder.py` and determines the object location, orientation,
and linear speed
* transform.ppy: contains utility functions for conducting transformations between coordinate frames
and extracting information about relative translation and orientation
*
*
*

## Setup and Execution
(in addition to tf, get numpy and scikit-learn?)
Set up world:
```
```
To run the motion detector:
```
python finder.py
```

To move the other robot:
```
python random_walk.py
```

#