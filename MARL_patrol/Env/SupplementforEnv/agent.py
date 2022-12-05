# -*- coding: utf-8 -*-
# author Amrish Bakaran
# author Adheesh
# author Bala Murali
# Copyright
# brief Drone class containing state, methods and history of the drone

import sys

import numpy as np

sys.path.append("..")

from Env.SupplementforEnv.constants import CONSTANTS as K

CONST = K()
np.set_printoptions(precision=3, suppress=True)


class Agent:
    def __init__(self, x=25 + CONST.GRID_SZ / 2, y=25 + CONST.GRID_SZ / 2):
        self.curPos = np.array([x, y])
        self.curVel = np.array([0, 0])
        self.size = 0.4
        self.tourTaken = []  # list of positions that the robot has taken
        self.maxVelocity = CONST.MAX_AGENT_VEL  # m/s

    def setParams(self, vel):
        self.curVel = vel * self.maxVelocity

    def updateState(self, timeStep):
        self.updatePos(timeStep)
        self.updateTour()

    def predNewState(self, timeStep):
        newPosition = self.curPos + self.curVel * timeStep
        return np.round(newPosition, 3)

    def updatePos(self, timeStep):
        newPosition = self.curPos + self.curVel * timeStep
        self.curPos = np.round(newPosition, 3)

    #        print("Drone POS = ", self.curPos)

    def updateTour(self):
        if len(self.tourTaken) > 0:
            # print((self.tourTaken[-1] == self.curPos).all())
            if not (self.tourTaken[-1] == self.curPos).all():
                self.tourTaken.append(self.curPos)
                # print(len(self.tourTaken))
        else:
            self.tourTaken.append(self.curPos)

    def getState(self):
        return self.curPos, self.curVel, self.tourTaken
