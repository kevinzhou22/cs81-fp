#!/usr/bin/env python

# Author: Muhtasim Miraz
# Date: 05/17/2023

import numpy as np
from sklearn.cluster import DBSCAN

class MovingObjectDetector:
    def __init__(self, eps=1.5, min_samples=4):
        
        # maximum distance between two samples for them to be considered as in the same neighborhood
        # default is 1.5 grid units
        self.eps = eps

        # minimum number of samples in a neighborhood for a point to be considered as a core point
        # default is 4 points 
        self.min_samples = min_samples

    def detect_object(self, points):
        points = np.array(points)
        if len(points) == 0:
            return None

        # cluster the points using DBSCAN
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
        labels = db.labels_
        clusters = [points[labels == i] for i in range(max(labels) + 1)]

        # filter out clusters that are too small
        clusters = [cluster for cluster in clusters if len(cluster) >= self.min_samples]
        print("clusters:\n", clusters)

        if not clusters:
            return None
        
        # find the largest cluster i.e. assume object is the largest cluster
        largest_cluster = max(clusters, key=lambda cluster: len(cluster))

        # compute the center of mass of that cluster
        centroid = np.mean(largest_cluster, axis=0)

        print("centroid:\n", centroid)

        return centroid

# if __name__ == "__main__":
    
#     detector = MovingObjectDetector(eps=1.5, min_samples=4)
#     # replace with points from the mapper script
#     # points1 = [(-3.1999737666070804, -2.4117316917632117),
#     #            (-3.1999732640835399, -2.4558711162253291),
#     #            (-3.0999737082086924, -2.4224627244005483),
#     #            (-3.0999731358005618, -2.4663961341470961),
#     #            (-2.9999734520330654, -2.42994652726428),
#     #            (-2.999972966809568, -2.4736728115017024),
#     #            (-2.9999722547759413, -2.5180336433110719),
#     #            (-2.9999718299326368, -2.5630502994602891),
#     #            (-2.9999711994814082, -2.6087440363303434),
#     #            (-2.9999706546459981, -2.6551376412511662),
#     #            (-3.0999690253054202, -2.7923295633076393)]
    # centroid1 = detector.detect_object(anomalies)
    # # should print the (x, y) coordinates of the center of the object
    # print("Object location:", centroid1, "\n")

# # example points
# # no noise, only one object
# points1 = [
#     [1.0, 1.0],
#     [1.5, 1.5],
#     [2.0, 2.0],
#     [2.5, 2.5],
#     [3.0, 3.0],
#     [3.5, 3.5],
# ]

# # one object + noise
# points2 = [
#     [1.0, 1.0],
#     [1.5, 1.5],
#     [2.0, 2.0],
#     [2.5, 2.5],
#     [3.0, 3.0],
#     [10.0, 10.0],  # Noise point
#     [11.0, 11.0],  # Noise point
#     [12.0, 12.0],  # Noise point
# ]

# # one object + one other cluster + noise
# points3 = [
#     [1.0, 1.0],  # Noise point
#     [2.0, 2.0],  # Noise point
#     [5.0, 5.0], # cluster 1
#     [5.5, 5.5],
#     [6.0, 6.0],
#     [6.5, 6.5],
#     [10.0, 10.0],  # cluster 2
#     [10.5, 10.5],  
#     [11.0, 11.0],  
#     [11.5, 11.5],  
#     [12.0, 12.0], 
#     [12.5, 12.5],  
# ]

# centroid1 = detector.detect_object(points1)
# # should print the (x, y) coordinates of the center of the object
# print("Object 1:", centroid1, "\n")

# centroid2 = detector.detect_object(points2)
# print("Object 2:", centroid2, "\n")


# centroid3 = detector.detect_object(points3)
# print("Object 3:", centroid3, "\n")