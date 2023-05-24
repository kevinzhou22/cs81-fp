#!/usr/bin/env python

# Author: Amel Docena
# Date: May 23, 2023

"""
Credits to Kizito Masaba for sharing sample codes for killing nodes
TODO: Where are the codes to reset the simulation though?

"""

import os
import signal
import socket
import subprocess
import sys
import time
import sys
import numpy
import rospy



"""
Running nodes: World file
    
Killing nodes
Reset the simulation
"""



def check_kill_process(pstring):
    for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        print(line)
        try:
            os.kill(int(pid), signal.SIGKILL)
        except OSError:
            print("No process")


def start_simulation(launcher_args):
    rospy.logerr(launcher_args)
    print(launcher_args)
    main_process = subprocess.Popen(launcher_args)
    main_process.wait()
    check_kill_process("ros")
    time.sleep(10)

def restart_simulation():
    pass