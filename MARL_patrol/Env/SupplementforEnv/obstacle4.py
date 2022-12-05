# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:14:44 2020

@author: amris
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:14:44 2020

@author: amris
"""
import math
import sys

import numpy as np

sys.path.append("..")
from Env.SupplementforEnv.constants import CONSTANTS as K
from matplotlib.path import Path
from functools import partial
from collections import defaultdict
from heapq import *

CONST = K()

import time

from Env.SupplementforEnv.vsb2 import Visibility


class Obstacle:
    def __init__(self):

        self.nodes = {0: [5, 40], 1: [5, 35], 2: [5, 30], 3: [5, 25], 4: [5, 20], 5: [5, 15], 6: [5, 10], 7: [5, 5],
                      8: [5, 0], 9: [10, 0], 10: [15, 0], 11: [20, 0], 12: [25, 0], 13: [30, 0], 14: [35, 0],
                      15: [40, 0], 16: [40, 5],
                      17: [40, 10], 18: [40, 15], 19: [40, 20], 20: [40, 25], 21: [40, 30], 22: [40, 35], 23: [40, 40]}
        self.patrol_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    def getAllObs_vsbs(self, emptyMap):
        obsMaps = []
        vsbs = []
        vsbPolys = []
        numOpenCellsArr = []
        a = time.time()

        #        mp, vsb = self.getObstacleMap(emptyMap, self.obstacle1())
        #        obsMaps.append(mp)
        #        vsbs.append(vsb)
        #        vsbPoly =  self.getVisibilityPolys(vsb, mp)
        #        vsbPolys.append(vsbPoly)
        #        numOpenCellsArr.append(np.count_nonzero(mp==0))
        #
        #        mp, vsb = self.getObstacleMap(emptyMap, self.obstacle2())
        #        obsMaps.append(mp)
        #        vsbs.append(vsb)
        #        vsbPoly =  self.getVisibilityPolys(vsb, mp)
        #        vsbPolys.append(vsbPoly)
        #        numOpenCellsArr.append(np.count_nonzero(mp==0))
        #
        #        mp, vsb = self.getObstacleMap(emptyMap, self.obstacle3())
        #        obsMaps.append(mp)
        #        vsbs.append(vsb)
        #        vsbPoly =  self.getVisibilityPolys(vsb, mp)
        #        vsbPolys.append(vsbPoly)
        #        numOpenCellsArr.append(np.count_nonzero(mp==0))
        #
        #        mp, vsb = self.getObstacleMap(emptyMap, self.obstacle4())
        #        obsMaps.append(mp)
        #        vsbs.append(vsb)
        #        vsbPoly =  self.getVisibilityPolys(vsb, mp)
        #        vsbPolys.append(vsbPoly)
        #        numOpenCellsArr.append(np.count_nonzero(mp==0))

        mp, vsb = self.getObstacleMap(emptyMap, self.obstacle6())
        obsMaps.append(mp)
        vsbs.append(vsb)
        vsbPoly = self.getVisibilityPolys(vsb, mp)
        vsbPolys.append(vsbPoly)
        numOpenCellsArr.append(np.count_nonzero(mp == 0))
        self.get_allnodes_matrix()
        b = time.time()
        print("create vsb Polys:", round(1000 * (b - a), 3))
        return obsMaps, vsbs, vsbPolys, numOpenCellsArr

    def getObstacleMap(self, emptyMap, obstacleSet):
        obsList = obstacleSet
        vsb = Visibility(emptyMap.shape[0], emptyMap.shape[1])
        for obs, isHole in obsList:
            vsb.addGeom2Arrangement(obs)

        isHoles = [obs[1] for obs in obsList]
        if any(isHoles) == True:
            pass
        else:
            vsb.boundary2Arrangement(vsb.length, vsb.height)

        # get obstacle polygon
        points = CONST.GRID_CENTER_PTS
        img = np.zeros_like(emptyMap, dtype=bool)
        for obs, isHole in obsList:
            p = Path(obs)
            grid = p.contains_points(points)
            mask = grid.reshape(CONST.MAP_SIZE, CONST.MAP_SIZE)
            img = np.logical_or(img, (mask if not isHole else np.logical_not(mask)))

        img = img.T
        img = np.where(img, 150, emptyMap)
        return img, vsb

    def getVisibilityPolys(self, vsb, obsMap):
        polys = defaultdict(partial(np.ndarray, 0))
        for pt in CONST.GRID_CENTER_PTS:
            if not obsMap[int(pt[0]), int(pt[1])] == 150:
                polys[(pt[0], pt[1])] = vsb.getVsbPoly(pt)

        return polys

    def getObstacles(self):
        obstacle = self.obstacle1()
        return obstacle

    def get_allnodes_matrix(self):

        neighbor_matrix = np.zeros((24, 24), np.int32)
        neighbor_matrix[0, 1] = 5
        neighbor_matrix[1, 2] = 5
        neighbor_matrix[2, 3] = 5
        neighbor_matrix[3, 4] = 5
        neighbor_matrix[4, 5] = 5
        neighbor_matrix[5, 6] = 5
        neighbor_matrix[6, 7] = 5
        neighbor_matrix[7, 8] = 5
        neighbor_matrix[8, 9] = 5
        neighbor_matrix[9, 10] = 5
        neighbor_matrix[10, 11] = 5
        neighbor_matrix[11, 12] = 5
        neighbor_matrix[12, 13] = 5
        neighbor_matrix[13, 14] = 5
        neighbor_matrix[14, 15] = 5
        neighbor_matrix[15, 16] = 5
        neighbor_matrix[16, 17] = 5
        neighbor_matrix[17, 18] = 5
        neighbor_matrix[18, 19] = 5
        neighbor_matrix[19, 20] = 5
        neighbor_matrix[20, 21] = 5
        neighbor_matrix[21, 22] = 5
        neighbor_matrix[22, 23] = 5

        self.neighbor_matrix = neighbor_matrix + neighbor_matrix.T
        self.neighbor_matrix = np.where(self.neighbor_matrix == 0, math.inf, self.neighbor_matrix)
        for i in range(np.size(self.neighbor_matrix, 0)):
            self.neighbor_matrix[i][i] = 0

        self.edges = []
        for i in range(np.size(self.neighbor_matrix, 0)):
            for j in range(np.size(self.neighbor_matrix, 1)):
                if i != j and self.neighbor_matrix[i][j] != math.inf:
                    self.edges.append(
                        (i, j, self.neighbor_matrix[i][
                            j]))  ### (i,j) is a link; M_topo[i][j] here is 1, the length of link (i,j).
        adj = np.zeros((24, 24), dtype=int)

        adj = np.where(self.neighbor_matrix == 5, 1, adj)
        self.adj = adj

    def get_key(self, val, my_dict):
        for key, value in my_dict.items():
            if val[0] == value[0] and val[1] == value[1]:
                return key

    def get_neighbor_ptnodes(self, node_pos):
        neighbor_nodes = []

        ptnode = self.get_key(node_pos, self.nodes)
        idx = np.nonzero(self.adj[int(ptnode)])
        for item in idx[0]:
            neighbor_nodes.append(item)

        nb_nodes = []
        for i in range(len(neighbor_nodes)):
            nb_nodes.append(self.nodes[neighbor_nodes[i]])
        return nb_nodes

    def get_neighbor_nodes_number(self, ptnode):
        neighbor_nodes = []

        idx = np.nonzero(self.adj[int(ptnode)])
        for item in idx[0]:
            neighbor_nodes.append(item)

        return neighbor_nodes

    def dijkstra_raw(self, from_node, to_node):
        g = defaultdict(list)
        for l, r, c in self.edges:
            g[l].append((c, r))
        q, seen = [(0, from_node, ())], set()
        while q:
            (cost, v1, path) = heappop(q)
            if v1 not in seen:
                seen.add(v1)
                path = (v1, path)
                if v1 == to_node:
                    return cost, path
                for c, v2 in g.get(v1, ()):
                    if v2 not in seen:
                        heappush(q, (cost + c, v2, path))
        return float("inf"), []

    def dijkstra(self, from_node, to_node):
        len_shortest_path = -1
        ret_path = []
        length, path_queue = self.dijkstra_raw(from_node, to_node)
        if len(path_queue) > 0:
            len_shortest_path = length  ## 1. Get the length firstly;
            ## 2. Decompose the path_queue, to get the passing nodes in the shortest path.
            left = path_queue[0]
            ret_path.append(left)  ## 2.1 Record the destination node firstly;
            right = path_queue[1]
            while len(right) > 0:
                left = right[0]
                ret_path.append(left)  ## 2.2 Record other nodes, till the source-node.
                right = right[1]
            ret_path.reverse()  ## 3. Reverse the list finally, to make it be normal sequence.
        return len_shortest_path, ret_path

    def patrol_pts(self):
        pt0 = []
        for i in range(len(self.patrol_nodes)):
            pt0.append(self.nodes[self.patrol_nodes[i]][0])
        pt1 = []
        for i in range(len(self.patrol_nodes)):
            pt1.append(self.nodes[self.patrol_nodes[i]][1])
        pt = pt0, pt1

        return pt

    def obstacle6(self):
        obsList = []
        # add points in CW order and
        isHole = False
        # geom = [[5,0],[5,40],[6,40],[6,1],[40,1],[40,40],[41,40],[41,0],[50,0],[50,50],[0,50],[0,0]]
        geom = [[0, 0], [0, 50], [50, 50], [50, 0], [41, 0], [41, 41], [40, 41], [40, 1], [6, 1], [6, 41], [5, 41],
                [5, 0]]
        obsList.append([geom, isHole])

        return obsList


obsMap = Obstacle()
obsMaps, vsbs, vsbPolys, numOpenCellsArr = obsMap.getAllObs_vsbs(np.zeros((CONST.MAP_SIZE, CONST.MAP_SIZE)))
