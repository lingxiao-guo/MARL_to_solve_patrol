# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 01:33:51 2020

@author: amris
"""
import time
import copy
import skgeom as sg
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
import sys
sys.path.append("..")

from Env.SupplementforEnv.constants import CONSTANTS as K

CONST = K()


class Visibility:
    def __init__(self, length, height):
        self.length = length
        self.height = height
        self.boundary = sg.arrangement.Arrangement()
        self.boundary_obs = sg.arrangement.Arrangement()
        self.visibilityPolygon = None
        #        self.boundary2Arrangement(length,height)
        self.obsPolyList = []

    def addGeom2Arrangement(self, pts):
        tempObs = sg.arrangement.Arrangement()
        # pts in 2D list
        edges = self.pts2Edges(pts)
        for ed in edges:
            tempObs.insert(ed)
            self.boundary_obs.insert(ed)
        #        self.obsPolyList.append(self.getSgPolyFromArr(tempObs))
        for he in tempObs.halfedges:
            sg.draw.draw(he.curve(), visible_point=False)

    def boundary2Arrangement(self, length, height):
        pts = [[0, 0],
               [0, height],
               [length, height],
               [length, 0]]
        edges = self.pts2Edges(pts)
        for ed in edges:
            self.boundary.insert(ed)
            self.boundary_obs.insert(ed)
        for he in self.boundary.halfedges:
            sg.draw.draw(he.curve(), color='red', visible_point=False)

    def pts2Edges(self, pts):
        edges = []
        for i in range(1, len(pts)):
            e = sg.Segment2(sg.Point2(pts[i - 1][0], pts[i - 1][1]),
                            sg.Point2(pts[i][0], pts[i][1]))
            edges.append(e)
        e = sg.Segment2(sg.Point2(pts[len(pts) - 1][0], pts[len(pts) - 1][1]), sg.Point2(pts[0][0], pts[0][1]))
        edges.append(e)
        return edges

    def getSgPolyFromArr(self, arr):
        allEdges = []
        for e in arr.halfedges:
            edge = [e.source().point(), e.target().point()]
            edgeRev = [e.target().point(), e.source().point()]
            if not edgeRev in allEdges and not edge in allEdges:
                allEdges.append([e.source().point(), e.target().point()])

        polyPts = []
        prevEndPt = allEdges[0][1]
        polyPts.append(allEdges[0][0])
        allEdges.pop(0)

        while len(allEdges) > 0:
            #            print(len(allEdges))
            for e in allEdges:
                if prevEndPt == e[0]:
                    polyPts.append(prevEndPt)
                    prevEndPt = e[1]
                    allEdges.remove(e)
                elif prevEndPt == e[1]:
                    polyPts.append(prevEndPt)
                    prevEndPt = e[0]
                    allEdges.remove(e)
                    break
        poly = sg.Polygon(polyPts)
        return poly

    def getVisibilityPolygon(self, fromPt):
        vs = sg.RotationalSweepVisibility(self.boundary_obs)
        q = sg.Point2(fromPt[0], fromPt[1])
        face = self.boundary_obs.find(q)
        vx = vs.compute_visibility(q, face)
        visibilityPolygon = self.getSgPolyFromArr(vx)
        self.visbilityPolygon = visibilityPolygon
        return visibilityPolygon

    def isPtinPoly(self, pt, polygon):
        # pt as a list
        position = polygon.oriented_side(sg.Point2(pt[0], pt[1]))
        if position == sg.Sign.POSITIVE:
            return 1
        elif position == sg.Sign.NEGATIVE:
            return -1
        elif position == sg.Sign.ZERO:
            return 0

    def updateVsbPolyOnImgOld(self, pt, img):
        times = []
        a = time.time()
        # update for multiple agents
        vsbPoly = self.getVisibilityPolygon(pt[0])
        #        print(pt[0])
        b = time.time()
        times.append(["getVsbPoly", round(1000 * (b - a), 3)])
        # =============================================================================
        #         # use to check if polygon is wrong
        #         if pt[0][0] == 0.5 and pt[0][1] == 10.5:
        #             sg.draw.draw_polygon(vsbPoly)
        #             plt.pause(0.01)
        # =============================================================================

        a = time.time()
        lib_insideID = self.isPtinPoly(pt[0], vsbPoly)
        b = time.time()
        times.append(["checkPtInPoly", round(1000 * (b - a), 3)])

        a = time.time()
        for i in range(0, int(CONST.MAP_SIZE)):
            for j in range(0, int(CONST.MAP_SIZE)):
                x = CONST.GRID_SZ / 2 + i * CONST.GRID_SZ
                y = CONST.GRID_SZ / 2 + j * CONST.GRID_SZ
                if self.isPtinPoly([x, y], vsbPoly) == lib_insideID:
                    img[i, j] = 255
        b = time.time()
        times.append(["fillMap", round(1000 * (b - a), 3)])

        print(times)
        return img

    def updateVsbPolyOnImg(self, pt, img):
        # update for multiple agents
        vsbPoly = self.getVisibilityPolygon(pt[0])
        #        print(pt[0])

        points = CONST.GRID_CENTER_PTS
        p = Path(vsbPoly.coords)
        grid = p.contains_points(points)
        mask = grid.reshape(50, 50)
        vsbGrid = mask.T
        temp = np.copy(img)
        # updating grid with matplotlib calculated visibility grid
        temp = np.where(vsbGrid, 255, temp)
        # adding the obstacles back
        #        temp = np.where(img == 150, 150, temp)
        return temp

    def getVsbPoly(self, pt):
        vsbPoly = self.getVisibilityPolygon(pt)
        return vsbPoly.coords

    def checkPtInVsbPoly(self, pt, checkPt):
        vsbPoly = self.getVisibilityPolygon(pt[0])

        points = [checkPt[0]]
        p = Path(vsbPoly.coords)
        canSee = p.contains_points(points)
        return canSee

    def checkPtInVsbPolyDict(self, pt, checkPt, vsbPolyDict):

        points = [checkPt[0]]
        p = pt[0]
        p = Path(vsbPolyDict[(p[0], p[1])])
        canSee = p.contains_points(points)
        return canSee

    def updateVsbOnImg(self, pt, gPt, img, vsbPolyDict):
        p = pt[0]
        g = gPt[0]
        p = Path(vsbPolyDict[(p[0], p[1])])
        points = CONST.GRID_CENTER_PTS
        grid = p.contains_points(points)
        mask = grid.reshape(CONST.MAP_SIZE, CONST.MAP_SIZE)
        # side of local visible area/2
        r = int((CONST.LOCAL_SZ - 1) / 2)
        lx = max(0, g[1] - r)
        hx = min(CONST.MAP_SIZE, r + g[1] + 1)
        ly = max(0, g[0] - r)
        hy = min(CONST.MAP_SIZE, r + g[0] + 1)
        tempMask = np.zeros_like(mask)
        tempMask[lx: hx, ly: hy] = 1

        mask = np.where(tempMask == 0, False, mask)
        vsbGrid = mask.T
        temp = np.copy(img)

        temp = np.where(vsbGrid, 255, temp)

        return temp


    def update_visibility_get_local(self, pt, gPt, img, vsbPolyDict, patrol_pts):
        p = pt
        g = gPt
        vertice=vsbPolyDict[(p[0], p[1])]
        p = Path(vertice)
        points = CONST.GRID_CENTER_PTS
        grid = p.contains_points(points)
        mask = grid.reshape(CONST.MAP_SIZE, CONST.MAP_SIZE)
        # side of local visible area/2
        r = 0   #int((CONST.LOCAL_SZ - 1) / 2)
        lx = int(max(0, g[1] - r))
        hx = int(min(CONST.MAP_SIZE, r + g[1] + 1))
        ly = int(max(0, g[0] - r))
        hy = int(min(CONST.MAP_SIZE, r + g[0] + 1))
        tempMask = np.zeros_like(mask)
        tempMask[lx: hx, ly: hy] = 1

        mask = np.where(tempMask == 0, False, mask)
        vsbGrid = mask.T

        # try
        temp = img
        temp[patrol_pts] = np.where(vsbGrid[patrol_pts], 255, img[patrol_pts])
        #temp = np.where(vsbGrid, 255, img)

        return temp

