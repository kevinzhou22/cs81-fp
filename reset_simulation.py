#!/usr/bin/env python

# Author: Amel Docena
# Date: May 23, 2023

"""
Credits to Kizito Masaba for sharing sample codes for killing nodes
TODO: Where are the codes to reset the simulation though?

"""

import os
#import rospy

def check_kill_process(self, pstring):
    for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        print(line)
        os.kill(int(pid), signal.SIGKILL)

def kill_ros(self):
    sleep(10)
    self.check_kill_process(self.method)
    all_nodes = []
    # count = self.robot_count
    for i in range(self.robot_count):
        all_nodes += ['/robot_{}/GetMap'.format(i),
                      '/robot_{}/navigator'.format(i), '/robot_{}/operator'.format(i),
                      '/robot_{}/robot_nav'.format(i),
                      '/robot_{}/SetGoal'.format(i), '/robot_{}/fake_localization'.format(i)]
        # all_nodes += ['/robot_{}/explore_client'.format(i), '/robot_{}/graph'.format(i),'/robot_{}/{}'.format(i,self.method)]

    all_nodes += ['/rosout', '/RVIZ', '/Stage', '/map_server', '/roscbt', '/sensor_simulator']
    rosnode.kill_nodes(all_nodes)
    rospy.signal_shutdown("Sampling complete! Shutting down")

def reset_simulation():
    pass