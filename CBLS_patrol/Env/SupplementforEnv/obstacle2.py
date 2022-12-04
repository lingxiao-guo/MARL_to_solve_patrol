# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:14:44 2020

@author: amris
"""
import math

import skgeom as sg
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("..")
from Env.SupplementforEnv.constants import CONSTANTS as K
from matplotlib.path import Path
from collections import defaultdict
from functools import partial
from collections import defaultdict
from heapq import *
CONST = K()

import time

from Env.SupplementforEnv.vsb2 import Visibility


class Obstacle:
    def __init__(self):
        self.nodes = {0: [6, 0], 1: [20, 0], 2: [28, 0], 3: [37, 0], 4: [46, 0], 5: [8, 15], 6: [13, 12], 7: [20, 12],
                      8: [32, 10], 9: [37, 10], 10: [4, 25], 11: [8, 25],
                      12: [13, 25], 13: [19, 25], 14: [19, 21], 15: [28, 21], 16: [38, 21], 17: [46, 21], 18: [46, 30],
                      19: [38, 30], 20: [35, 32], 21: [29, 32], 22: [35, 25],
                      23: [22, 25], 24: [22, 32], 25: [9, 32], 26: [0, 32], 27: [4, 32], 28: [0, 40], 29: [9, 42],
                      30: [19, 40], 31: [29, 40], 32: [38, 40], 33: [46, 40]}

        self.patrol_nodes = [0, 1, 2, 3, 5, 6, 8, 12, 15, 16, 17, 18, 19, 20, 23, 27, 28, 29, 30, 31]

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


        neighbor_matrix=np.zeros((34,34),np.int32)
        neighbor_matrix[0,1]=14
        neighbor_matrix[1,2]=8
        neighbor_matrix[1, 7] = 8
        neighbor_matrix[2, 3] = 9
        neighbor_matrix[2, 15] = 21
        neighbor_matrix[3, 9] = 10
        neighbor_matrix[8,9] = 5
        neighbor_matrix[3, 4] = 9
        neighbor_matrix[4, 17] = 21
        neighbor_matrix[16, 17] = 8
        neighbor_matrix[16, 19] = 9
        neighbor_matrix[15, 16] = 10
        neighbor_matrix[14, 15] = 9
        neighbor_matrix[13, 14] = 4
        neighbor_matrix[12, 13] = 6
        neighbor_matrix[11, 12] = 5
        neighbor_matrix[6, 12] = 13
        neighbor_matrix[6, 7] = 7
        neighbor_matrix[5, 11] = 10
        neighbor_matrix[10, 11] = 4
        neighbor_matrix[10, 27] =7
        neighbor_matrix[26, 27] = 4
        neighbor_matrix[25, 27] = 5
        neighbor_matrix[26, 28] = 8
        neighbor_matrix[25, 29] = 10
        neighbor_matrix[13, 30] = 15
        neighbor_matrix[30, 31] = 10
        neighbor_matrix[21, 31] = 8
        neighbor_matrix[21, 24] = 7
        neighbor_matrix[23, 24] = 7
        neighbor_matrix[20, 21] = 6
        neighbor_matrix[20, 22] = 3
        neighbor_matrix[22, 23] = 13
        neighbor_matrix[31, 32] = 9
        neighbor_matrix[19, 32] = 10
        neighbor_matrix[18, 19] = 8
        neighbor_matrix[17, 18] = 9
        neighbor_matrix[18, 33] = 10
        neighbor_matrix[32, 33] = 8

        self.neighbor_matrix=neighbor_matrix+neighbor_matrix.T
        self.neighbor_matrix=np.where(self.neighbor_matrix==0,math.inf,self.neighbor_matrix)
        for i in range(np.size(self.neighbor_matrix,0)):
            self.neighbor_matrix[i][i]=0

        self.edges = []
        for i in range(np.size(self.neighbor_matrix,0)):
            for j in range(np.size(self.neighbor_matrix,1)):
                if i != j and self.neighbor_matrix[i][j] != math.inf:
                    self.edges.append(
                        (i, j, self.neighbor_matrix[i][j]))
        adj=np.zeros((20,20),dtype=int)
        adj[0][1]=1
        adj[1][2]=1
        adj[1][5]=1
        adj[2][3]=1
        adj[2][8]=1
        adj[3][6]=1
        adj[3][10]=1
        adj[4][7]=1
        adj[4][15]=1
        adj[5][7]=1
        adj[7][15]=1
        adj[7][8]=1
        adj[7][18]=1
        adj[8][9]=1
        adj[8][18]=1
        adj[9][10]=1
        adj[9][12]=1
        adj[10][11]=1
        adj[11][12]=1
        adj[11][19]=1
        adj[12][19]=1
        adj[13][14]=1
        adj[13][19]=1
        adj[14][19]=1
        adj[15][16]=1
        adj[15][17]=1
        adj[18][19]=1
        self.adj=adj+adj.T




    def get_key(self,val,my_dict):
        for key, value in my_dict.items():
            if val[0]==value[0] and val[1]==value[1]:
                return key

    def get_neighbor_ptnodes(self,node_pos):
        neighbor_nodes=[]

        ptnode=self.get_key(node_pos,self.nodes)

        if ptnode==0:
            neighbor_nodes=[1]
        elif ptnode==1:
            neighbor_nodes=[0,2,6]
        elif ptnode==2:
            neighbor_nodes=[1,3,15]
        elif ptnode==3:
            neighbor_nodes=[2,8,17]
        elif ptnode==5:
            neighbor_nodes=[27,12]
        elif ptnode==6:
            neighbor_nodes=[12,1]
        elif ptnode==8:
            neighbor_nodes=[3]
        elif ptnode==12:
            neighbor_nodes=[27,5,6,30,15]
        elif ptnode==15:
            neighbor_nodes=[2,12,16,30]
        elif ptnode==16:
            neighbor_nodes=[15,19,17]
        elif ptnode==17:
            neighbor_nodes=[16,18,3]
        elif ptnode==18:
            neighbor_nodes=[17,19,31]
        elif ptnode==19:
            neighbor_nodes=[16,18,31]
        elif ptnode==20:
            neighbor_nodes=[23,31]
        elif ptnode==23:
            neighbor_nodes=[20,31]
        elif ptnode==27:
            neighbor_nodes=[28,29,6,12]
        elif ptnode==28:
            neighbor_nodes=[27]
        elif ptnode==29:
            neighbor_nodes=[27]
        elif ptnode==30:
            neighbor_nodes=[12,15,31]
        elif ptnode==31:
            neighbor_nodes=[30,20,23,19,18]
        else:
            raise Exception("it's not a patrolling point!")
        nb_nodes=[]
        for i in range(len(neighbor_nodes)) :
            nb_nodes.append(self.nodes[neighbor_nodes[i]])
        return nb_nodes

    def get_neighbor_nodes_number(self,ptnode):
        neighbor_nodes = []

        if ptnode == 0:
            neighbor_nodes = [1]
        elif ptnode == 1:
            neighbor_nodes = [0, 2, 6]
        elif ptnode == 2:
            neighbor_nodes = [1, 3, 15]
        elif ptnode == 3:
            neighbor_nodes = [2, 8, 17]
        elif ptnode == 5:
            neighbor_nodes = [27, 12]
        elif ptnode == 6:
            neighbor_nodes = [12, 1]
        elif ptnode == 8:
            neighbor_nodes = [3]
        elif ptnode == 12:
            neighbor_nodes = [27, 5, 6, 30, 15]
        elif ptnode == 15:
            neighbor_nodes = [2, 12, 16, 30]
        elif ptnode == 16:
            neighbor_nodes = [15, 19, 17]
        elif ptnode == 17:
            neighbor_nodes = [16, 18, 3]
        elif ptnode == 18:
            neighbor_nodes = [17, 19, 31]
        elif ptnode == 19:
            neighbor_nodes = [16, 18, 31]
        elif ptnode == 20:
            neighbor_nodes = [23, 31]
        elif ptnode == 23:
            neighbor_nodes = [20, 31]
        elif ptnode == 27:
            neighbor_nodes = [28, 29, 6, 12]
        elif ptnode == 28:
            neighbor_nodes = [27]
        elif ptnode == 29:
            neighbor_nodes = [27]
        elif ptnode == 30:
            neighbor_nodes = [12, 15, 31]
        elif ptnode == 31:
            neighbor_nodes = [30, 20, 23, 19, 18]
        else:
            raise Exception("it's not a patrolling point!")
        return neighbor_nodes




    def dijkstra_raw(self, from_node,to_node):
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
        length, path_queue = self.dijkstra_raw(from_node,to_node)
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


    def obstacle1(self):
        obsList = []
        # add points in CW order and
        isHole = False
        geom = [[7, 0],
                [7, 20],
                [29, 20],
                [29, 29],
                [10, 29],
                [10, 40],
                [30, 40],
                [30, 39],
                [11, 39],
                [11, 30],
                [30, 30],
                [30, 19],
                [8, 19],
                [8, 0]]
        obsList.append([geom, isHole])

        geom = [[10, 50],
                [10, 45],
                [25, 45],
                [25, 46],
                [11, 46],
                [11, 50]]
        obsList.append([geom, isHole])

        geom = [[5, 19],
                [5, 35],
                [6, 35],
                [6, 20],
                [7, 20],
                [7, 19]]
        obsList.append([geom, isHole])

        return obsList

    def obstacle2(self):
        obsList = []
        # add points in CW order and
        isHole = False
        geom = [[6, 6],
                [6, 12],
                [44, 12],
                [44, 6]]
        obsList.append([geom, isHole])

        geom = [[35, 18],
                [35, 23],
                [41, 23],
                [41, 18]]
        obsList.append([geom, isHole])

        geom = [[39, 29],
                [39, 34],
                [44, 34],
                [44, 29]]
        obsList.append([geom, isHole])

        geom = [[12, 29],
                [12, 39],
                [17, 39],
                [17, 29]]
        obsList.append([geom, isHole])

        geom = [[23, 39],
                [23, 44],
                [28, 44],
                [28, 39]]
        obsList.append([geom, isHole])

        return obsList

    def obstacle3(self):
        obsList = []
        # add points in CW order and
        isHole = True
        geom = [[0, 5],
                [0, 15],
                [3, 15],
                [3, 20],
                [0, 20],
                [0, 30],
                [3, 30],
                [3, 35],
                [0, 35],
                [0, 45],
                [9, 45],
                [6, 45],
                [6, 35],
                [6, 30],
                [15, 30],
                [15, 35],
                [12, 35],
                [12, 45],
                [21, 45],
                [21, 35],
                [18, 35],
                [18, 30],
                [27, 30],
                [27, 35],
                [24, 35],
                [24, 45],
                [33, 45],
                [33, 35],
                [30, 35],
                [30, 30],
                [39, 30],
                [39, 35],
                [36, 35],
                [36, 45],
                [50, 45],
                [50, 35],
                [42, 35],
                [42, 30],
                [50, 30],
                [50, 20],
                [42, 20],
                [42, 15],
                [50, 15],
                [50, 5],
                [36, 5],
                [36, 15],
                [39, 15],
                [39, 20],
                [30, 20],
                [30, 15],
                [33, 15],
                [33, 5],
                [24, 5],
                [24, 15],
                [27, 15],
                [27, 20],
                [18, 20],
                [18, 15],
                [21, 15],
                [21, 5],
                [12, 5],
                [12, 15],
                [15, 15],
                [15, 20],
                [6, 20],
                [6, 15],
                [9, 15],
                [9, 5]]
        obsList.append([geom, isHole])

        return obsList

    def obstacle_4R(self):
        obsList = []
        # add points in CW order and
        isHole = True
        geom = [
            [15, 35],
            [12, 35],
            [12, 45],
            [21, 45],
            [21, 35],
            [18, 35],
            [18, 30],
            [27, 30],
            [27, 35],
            [24, 35],
            [24, 45],
            [33, 45],
            [33, 35],
            [30, 35],

            [30, 20],
            [30, 15],
            [33, 15],
            [33, 5],
            [24, 5],
            [24, 15],
            [27, 15],
            [27, 20],
            [18, 20],
            [18, 15],
            [21, 15],
            [21, 5],
            [12, 5],
            [12, 15],
            [15, 15],
            [15, 20]
        ]
        obsList.append([geom, isHole])

        return obsList

    def obstacle4(self):
        obsList = []
        # add points in CW order and
        isHole = True
        geom = [[3, 30],
                [18, 30],
                [18, 12],
                [33, 12],
                [33, 27],
                [30, 27],
                [30, 30],
                [33, 30],
                [33, 39],
                [36, 39],
                [36, 33],
                [39, 33],
                [39, 36],
                [48, 36],
                [48, 27],
                [39, 27],
                [39, 30],
                [36, 30],
                [36, 21],
                [39, 21],
                [39, 24],
                [48, 24],
                [48, 15],
                [39, 15],
                [39, 18],
                [36, 18],
                [36, 12],
                [48, 12],
                [48, 9],
                [18, 9],
                [18, 6],
                [14, 6],
                [14, 27],
                [3, 27]]
        obsList.append([geom, isHole])

        return obsList

    def obstacle5(self):
        obsList = []
        # add points in CW order and
        isHole = True
        geom = [[6, 6],
                [6, 30],
                [44, 30],
                [44, 6],
                [42, 6],
                [42, 10],
                [40, 10],
                [40, 6]]
        obsList.append([geom, isHole])

        return obsList

    def obstacle2R(self):
        obsList = []
        # add points in CW order and
        isHole = True
        geom = [[15, 30],
                [18, 30],
                [30, 30],
                [30, 20],
                [30, 15],
                [33, 15],
                [33, 5],
                [24, 5],
                [24, 15],
                [27, 15],
                [27, 20],
                [18, 20],
                [18, 15],
                [21, 15],
                [21, 5],
                [12, 5],
                [12, 15],
                [15, 15],
                [15, 20],
                ]
        obsList.append([geom, isHole])

        return obsList

    def obstacle6(self):
        obsList = []
        # add points in CW order and
        isHole = False
        geom = [[6, 1],
                [6, 0],
                [0, 0],
                [0, 32],
                [4, 32],
                [4, 25],
                [8, 25],
                [8, 15],
                [9, 15],
                [9, 25],
                [13, 25],
                [13, 12],
                [20, 12],
                [20, 1]
                ]
        obsList.append([geom, isHole])

        geom = [[28, 1],
                [21, 1],
                [21, 13],
                [14, 13],
                [14, 25],
                [19, 25],
                [19, 21],
                [28, 21]
                ]
        obsList.append([geom, isHole])

        geom = [[19, 26],
                [5, 26],
                [5, 32],
                [10, 32],
                [10, 43],
                [9, 43],
                [9, 33],
                [1, 33],
                [1, 41],
                [0, 41],
                [0, 50],
                [50, 50],
                [50, 0],
                [47, 0],
                [47, 41],
                [19, 41]
                ]
        obsList.append([geom, isHole])

        geom = [[29, 33],
                [29, 40],
                [20, 40],
                [20, 22],
                [38, 22],
                [38, 40],
                [30, 40],
                [30, 33],
                [36, 33],
                [36, 25],
                [22, 25],
                [22, 33]]
        obsList.append([geom, isHole])

        geom = [[35, 32],
                [35, 26],
                [23, 26],
                [23, 32]]
        obsList.append([geom, isHole])

        geom = [[46, 40],
                [46, 31],
                [39, 31],
                [39, 40]]
        obsList.append([geom, isHole])

        geom = [[46, 30],
                [46, 22],
                [39, 22],
                [39, 30]]
        obsList.append([geom, isHole])

        geom = [[46, 21],
                [46, 1],
                [38, 1],
                [38, 11],
                [32, 11],
                [32, 10],
                [37, 10],
                [37, 1],
                [29, 1],
                [29, 21]]
        obsList.append([geom, isHole])
        return obsList

    def patrol_pts(self):
        pt0 = []
        for i in range(len(self.patrol_nodes)):
            pt0.append(self.nodes[self.patrol_nodes[i]][0])
        pt1 = []
        for i in range(len(self.patrol_nodes)):
            pt1.append(self.nodes[self.patrol_nodes[i]][1])
        pt = pt0, pt1

        return pt


obsMap = Obstacle()
obsMap.getAllObs_vsbs(np.zeros((CONST.MAP_SIZE, CONST.MAP_SIZE)))
#obsMaps, vsbs, vsbPolys, numOpenCellsArr = obsMap.getAllObs_vsbs(np.zeros((CONST.MAP_SIZE, CONST.MAP_SIZE)))

