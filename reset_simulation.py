#!/usr/bin/env python

# Author: Amel Docena
# Date: May 23, 2023

"""
Acknowledgements to Kizito Masaba for sharing sample codes for killing nodes, although these weren't used eventually.
"""

import os
import subprocess
import signal
import time
import rospy

"""
Running nodes: World file
    
Killing nodes
Reset the simulation
"""

def kill_nodes(nodes_to_kill, sleep):
    """
    Kills nodes on the nodes_to_kill list
    Args:
        nodes_to_kill:
        sleep:

    Returns:

    """
    for node in nodes_to_kill:
        for line in os.popen("ps ax | grep " + node + " | grep -v grep"):
            fields = line.split()
            pid = fields[0]
            try:
                print(line)
                os.kill(int(pid), signal.SIGKILL)
            except OSError:
                print("No process")
            time.sleep(sleep)

def launch_nodes(launch_file, sleep):
    """
    Runs the world file and list of python scripts
    Args:
        world: world file name
        python_scripts: list of python scripts

    Returns:

    """
    subprocess.Popen(['roslaunch', 'follower_dqn', launch_file])
    time.sleep(sleep)

def reset_simulation(launch_file, nodes_to_kill, sleep):
    """
    Kills nodes then launches the world and runs python scripts
    Args:
        path:
        nodes_to_kill:
        world:
        python_scripts:
        sleep:

    Returns:

    """
    kill_nodes(nodes_to_kill, sleep)
    launch_nodes(launch_file, sleep)