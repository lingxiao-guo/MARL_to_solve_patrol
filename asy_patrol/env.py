# -*- coding: utf-8 -*-
# author Amrish Bakaran
# Copyright
# brief Environment class for the simulation

import copy

import numpy as np
import random
import cv2
import time
import skimage.measure
from matplotlib import pyplot as plt



import sys
sys.path.append("..")
from SupplementforEnv.obstacle2 import Obstacle
obsMap = Obstacle()
from SupplementforEnv.agent import Agent

from SupplementforEnv.constants import CONSTANTS as K
CONST = K()

np.set_printoptions(precision=3, suppress=True)


class Env:
    timestep=0
    visits=[]
    value_list={}
    def __init__(self):
        self.timeStep = CONST.TIME_STEP
        # getting obstacle maps and visibility maps
        self.cap = CONST.TIME_STEP
        self.decay = 1
        self.patrol_points = obsMap.patrol_pts()
        self.avg_idleness=0
        self.obsMaps, self.vsbs, self.vsbPolys, self.numOpenCellsArr = self.initObsMaps_Vsbs()
        self.adj = obsMap.adj
        self.obstacle_map, self.vsb, self.vsbPoly, self.mapId, self.numOpenCells = self.setRandMap_vsb()
        # obstacle_map->obstacle_viewed->current_map_state
        # initialize environment with obstacles and agents
        agent_pos_list, self.obstacle_viewed, self.current_map_state, self.agents, self.agent_local_view_list = self.init_env(
            CONST.NUM_AGENTS, self.obstacle_map)
        # modified: decay rate:

        # modified: cap the upperbound of penalty

        # save video
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(f"checkpoints/cnn1.avi", self.fourcc, 50, (700, 700))

    def init_env(self, num_agents, obstacle_map):
        # unviewed = 0
        # viewed = 255
        # obstacle = 150
        # agent Pos = 100

        obstacle_viewed = np.copy(obstacle_map)

        #        obstacle_viewed = np.where(obstacle_viewed == 0, -1* self.cap, obstacle_viewed)

        # initialize agents at random location
        agents = []
        agents_local_view_list = []
        # get open locations
        #x, y = np.nonzero(obstacle_map == 0)
        #ndxs = random.sample(range(x.shape[0]), num_agents)
        ndxs=random.sample(range(len(obsMap.patrol_nodes)),num_agents)
        #ndxs=[0,8,16,18]
        #ndxs=[12,20]
        Env.visits = np.zeros(len(self.patrol_points[0]))
        for i in range(len(self.patrol_points[0])):
            Env.value_list[self.patrol_points[0][i], self.patrol_points[1][i]] = 0

        for ndx in ndxs[:num_agents]:
            agents.append(Agent(obsMap.nodes[obsMap.patrol_nodes[ndx]][0] + 0.5, obsMap.nodes[obsMap.patrol_nodes[ndx]][1] + 0.5))    #输入随机点的坐标。这里可以换一下，换成索引里点的坐标
        #            agents.append(Agent())

        agent_pos_list = [agent.getState()[0] for agent in agents]
        agent_g_pos_list = self.cartesian2grid(agent_pos_list)

        # update visibility for initial position
        for agent_pos, agent_g_pos in zip(agent_pos_list, agent_g_pos_list):
            obstacle_viewed = self.vsb.update_visibility_get_local(agent_pos, agent_g_pos, obstacle_viewed,
                                                                   self.vsbPoly,self.patrol_points)
        #            agents_local_view_list.append(temp)

        current_map_state = self.update_map_at_pos(agent_g_pos_list, obstacle_viewed,
                                                   100)
        self.local_heatmap_list = self.get_local_heatmap_list(current_map_state, agent_g_pos_list)

        self.mini_map = self.get_mini_map(current_map_state, 0.5, agent_g_pos_list)

        return agent_pos_list,obstacle_viewed, current_map_state, agents, agents_local_view_list

    def initObsMaps_Vsbs(self):
        return obsMap.getAllObs_vsbs(np.zeros((CONST.MAP_SIZE, CONST.MAP_SIZE)))

    def setRandMap_vsb(self):
        i = random.randint(0, len(self.obsMaps) - 1)
        return self.obsMaps[i], self.vsbs[i], self.vsbPolys[i], self.numOpenCellsArr[i], i

    def cartesian2grid(self, pos):
        g_pos = np.floor(pos).astype(np.int32)
        return g_pos

    def is_valid_location(self, pos, valid_map):
        g_pos = self.cartesian2grid(pos)
        # range check
        if 0 <= g_pos[0] < valid_map.shape[0] and 0 <= g_pos[1] < valid_map.shape[1]:
            pass
        else:
            return False
        return True
        # availability check
        #if valid_map[g_pos[0], g_pos[1]] == 0:  #
        #    return True
        #else:
        #   return False

    def update_map_at_pos(self, g_pos, use_map, val):
        updated_map = np.copy(use_map)

        for i in range(len(self.patrol_points[0])):
            updated_map[self.patrol_points[0][i],self.patrol_points[1][i]]=160
        #updated_map[self.patrol_points]=np.where(updated_map[self.patrol_points]==val,-self.decay,updated_map[self.patrol_points])
        updated_map=np.where(updated_map==val,0,updated_map)
        for pos in g_pos:
            updated_map[pos[0], pos[1]] = val
        return updated_map

    def step_agent(self, node_list,node_last_list,last_vertex_list):

        # have to decide on the action space
        # waypoints or velocity
        posOut = []
        velOut = []
        #list_of_key = list(obsMap.nodes.keys())
        #list_of_node_pos = list(obsMap.nodes.values())
       # last_vertex_list = list_of_key[list_of_node_pos.index(last_vertex_list)]
        last_vertex_list=[obsMap.get_key(last_vertex_list[i],obsMap.nodes) for i in range(CONST.NUM_AGENTS)]
        node_last_list = [obsMap.get_key(node_last_list[i], obsMap.nodes) for i in range(CONST.NUM_AGENTS)]
        node_list = [obsMap.get_key(node_list[i], obsMap.nodes) for i in range(CONST.NUM_AGENTS)]
        agent_pos_list = [agent.getState()[0] for agent in self.agents]
        agent_g_pos_list = self.cartesian2grid(agent_pos_list)
        for i,agent in zip(range(CONST.NUM_AGENTS),self.agents):
            vel = np.array([0, 0])
            length,route=obsMap.dijkstra(node_last_list[i],node_list[i])

            next_vertex=route[route.index(last_vertex_list[i])+1]
            next_vertex_pos=obsMap.nodes[next_vertex]
            if agent_g_pos_list[i][0]==next_vertex_pos[0] and agent_g_pos_list[i][1]<next_vertex_pos[1]:
                vel[1]=1
            elif agent_g_pos_list[i][0]==next_vertex_pos[0] and agent_g_pos_list[i][1]>next_vertex_pos[1]:
                vel[1] = -1
            elif agent_g_pos_list[i][0] < next_vertex_pos[0] and agent_g_pos_list[i][1] == next_vertex_pos[1]:
                vel[0] = 1
            elif agent_g_pos_list[i][0] > next_vertex_pos[0] and agent_g_pos_list[i][1] == next_vertex_pos[1]:
                vel[0]=-1

            agent.setParams(vel)
            predNewState = agent.predNewState(self.timeStep)
            # check if agent in obstacle
            agent_pos_list = [agent.getState()[0] for agent in self.agents]

            agent_g_pos_list = self.cartesian2grid(agent_pos_list)
           #  obstacle_map_with_agents = self.update_map_at_pos(agent_g_pos_list, self.obstacle_map, 100)
           # isValidPt = self.is_valid_location(predNewState, obstacle_map_with_agents)
            if True:  # if isValidPt
                agent.updateState(self.timeStep)
                curState = agent.getState()
                posOut.append(curState[0])
                velOut.append(curState[1])
           # else:
            #    curState = agent.getState()
             #   posOut.append(curState[0])
              #  velOut.append(curState[1])
            if (posOut[i][0]-0.5)==next_vertex_pos[0] and posOut[i][1]-0.5==next_vertex_pos[1]:
                last_vertex_list[i]=next_vertex
        last_v=[]
        for i in range(len(last_vertex_list)):
            last_v.append(obsMap.nodes[last_vertex_list[i]])

        return posOut, velOut,last_v




    def get_action_space(self):
        return [0, 1, 2, 3, 4]

    def step(self, node_list,node_last_list,last_vertex,state):

        Env.timestep+=1
        agent_pos_list, agent_vel_list,last_vertex_list= self.step_agent(node_list,node_last_list,last_vertex)
        agent_g_pos_list = self.cartesian2grid(agent_pos_list)

        for indx, agent_pos in enumerate(agent_pos_list):
            self.current_map_state = self.vsb.update_visibility_get_local(agent_pos, agent_g_pos_list[indx],
                                                                          self.current_map_state, self.vsbPoly,self.patrol_points)  #假设这个括号里的current_map_state就是只算巡逻点的

        #        # TODO local view with only visibility
        #        self.agent_local_view_list[indx]
        # Genearete currentStateMap with decay reward:

        # 150 is the wall, 255 (->0) (is newly viewed), initial unviewed is 0,  all pixels except wall is < 0, agent 100


        self.visit_flag=[False for _ in range(len(self.patrol_points[0]))]
        self.interval = np.zeros(len(self.patrol_points[0]))
        shared_reward=0

        for i in range(len(self.patrol_points[0])):
            visit_flag = False
            state[i][2]+=1
            rij=0
            if self.current_map_state[obsMap.nodes[obsMap.patrol_nodes[i]][0],obsMap.nodes[obsMap.patrol_nodes[i]][1]] == 255 :
                Env.visits[i]+=1
                visit_flag=True
                self.interval[i]=state[i][2]
                rij = self.reward(visit_flag, state[i][2])
                #rij=-10
                state[i][2]=0
                self.visit_flag[i]=True

            #rij=self.reward(Env.timestep,Env.visits[i],visit_flag)
            #rij=self.reward(i,visit_flag)
            #shared_reward+=rij
            Env.value_list[self.patrol_points[0][i],self.patrol_points[1][i]]+=rij
            self.current_map_state[self.patrol_points[0][i],self.patrol_points[1][i]]+=rij

        self.avg_idleness=np.sum(Env.timestep/(Env.visits+1))
        #shared_reward=shared_reward *(CONST.LEN_EPISODE-Env.timestep)/CONST.LEN_EPISODE-len(self.patrol_points[0])*Env.timestep/CONST.LEN_EPISODE
        # apply the lowerbound cap for the penalty

        state=np.array(state).squeeze()
        self.avg_i=np.mean(state[:,2]).squeeze()



        shared_reward=-self.avg_i/100+1

        #shared_reward=-(self.avg_i-self.avg_i_last)**2
        #shared_reward=-np.max(state[:,2]).squeeze()/100

        self.avg_i_last=self.avg_i
        self.avg_i_sum+=self.avg_i
        # update position to get current full map
        self.current_map_state = self.update_map_at_pos(agent_g_pos_list, self.current_map_state,
                                                        25)  # 更新一下地图里的智能体的位置信息

        self.local_heatmap_list = self.get_local_heatmap_list(self.current_map_state, agent_g_pos_list)

        self.mini_map = self.get_mini_map(self.current_map_state, 0.5, agent_g_pos_list)

#        local_reward_list, sr = self.get_reward(self.current_map_state)

        done = False
        flag=[]
        nodes_last_visit=[node_last_list[i] for i in range(CONST.NUM_AGENTS)]

        for i in range(CONST.NUM_AGENTS):
            if agent_g_pos_list[i][0]==node_list[i][0] and agent_g_pos_list[i][1]==node_list[i][1]:
                flag.append(True)
                nodes_last_visit[i]=node_list[i]
            else:
                flag.append(False)


        self.neighbor_nodes={}
        for i in range(CONST.NUM_AGENTS):
            self.neighbor_nodes[i]=obsMap.get_neighbor_ptnodes(nodes_last_visit[i])
        return state,agent_pos_list, self.current_map_state, self.neighbor_nodes,self.local_heatmap_list,\
               self.mini_map, shared_reward, flag,last_vertex_list,nodes_last_visit,self.visit_flag,done

    def reward(self,flag,idleness):
        rij=0
        if flag:
            rij=idleness
        return rij

    '''
    def reward(self,timestep,visits,flag):  
        if flag:
            rij = ((timestep - 1) ** 2 / (visits) - (timestep) ** 2 / (visits + 1))/2000
            return rij
        else:
            rij = -(2*timestep-1) / (visits + 1)/2000
            return rij
    '''
    '''
    def reward(self,i,visit_flag):
        if self.patrol_points[0][i]==28 and self.patrol_points[1][i]==21 and visit_flag==True:
            reward=1
        else:
            reward=0
        return reward
    '''
    def store_last(self):

        self.current_map_state_last=self.current_map_state.copy()
        self.visit_flag_copy=list(self.visit_flag)
        self.timestep_copy=Env.timestep
        self.visits_copy=Env.visits.copy()
        self.value_list_copy=dict(Env.value_list)
        self.agents_copy=copy.deepcopy(self.agents)

    def return_last(self):

        del self.current_map_state
        del self.visit_flag
        del Env.timestep
        del Env.visits
        del Env.value_list
        del self.agents

        self.current_map_state=self.current_map_state_last
        self.visit_flag=self.visit_flag_copy
        Env.timestep=self.timestep_copy
        Env.visits=self.visits_copy
        Env.value_list=self.value_list_copy
        self.agents=self.agents_copy

    def reset(self):

        # need to update initial state for reset function
        self.obstacle_map, self.vsb, self.vsbPoly, self.mapId, self.numOpenCells = self.setRandMap_vsb()
        agent_pos_list,self.obstacle_viewed, self.current_map_state, self.agents, self.agent_local_view_list = self.init_env(
            CONST.NUM_AGENTS, self.obstacle_map)
        #agent_pos_list=
        Env.visits = np.zeros(len(self.patrol_points[0]))
        self.avg_i=0
        self.avg_i_sum=0
        self.avg_i_last=0
        self.idleness_last_state=0
        Env.timestep=0
        self.avg_idleness=0
        self.worst_idleness=0
        self.worst_idleness_last=0
        self.interval=np.zeros(len(self.patrol_points[0]))

        state=[]
        for i in range(len(self.patrol_points[0])):
            Env.value_list[self.patrol_points[0][i], self.patrol_points[1][i]] = 0
            state.append([self.patrol_points[0][i], self.patrol_points[1][i],0])
        #action_list = [0 for _ in range(CONST.NUM_AGENTS)]
        #state = self.step(action_list)

        neighbor_nodes = {}

        nodes_last=[[agent_pos_list[i][0]-0.5,agent_pos_list[i][1] - 0.5] for i in range(CONST.NUM_AGENTS)]
        self.node_last_list=[]
        for i in range(CONST.NUM_AGENTS):
            self.node_last_list .append(obsMap.get_key([agent_pos_list[i][0] - 0.5, agent_pos_list[i][1] - 0.5], obsMap.nodes))

        last_vertex=[[agent_pos_list[i][0]-0.5,agent_pos_list[i][1] - 0.5] for i in range(CONST.NUM_AGENTS)]
        for i in range(CONST.NUM_AGENTS):
            neighbor_nodes[i]=obsMap.get_neighbor_ptnodes(nodes_last[i])
        flag=[True for i in range(CONST.NUM_AGENTS)]
        self.visit_flag=[False for i in range(len(obsMap.patrol_nodes))]
        return state,nodes_last,last_vertex,neighbor_nodes,flag


    def heatmap_render_prep(self, heatmap):
        cap = 200
        heatmapshow = np.rot90(heatmap, 1)

        heatmapshow = np.where(heatmapshow == 150, 1, heatmapshow)
        heatmapshow = np.where(heatmapshow == 200, 150, heatmapshow)
        heatmapshow = np.where(heatmapshow < 0, 0, -1 * heatmapshow)
        heatmapshow = np.where(heatmapshow >255 , 255, heatmapshow)

        heatmapshow = heatmapshow.astype(np.uint8)

        heatmapshow = cv2.applyColorMap(heatmapshow,11)
        return heatmapshow

    def exploration_render_prep(self, explr_map):
        img = np.copy(explr_map)
        img = np.rot90(img, 1)
        r = np.where(img == 150, 255, 0)
        g = np.where(img == 100, 255, 0)

        b = np.zeros_like(img)
        b_n = np.where(img == 255, 100, 0)
        bgr = np.stack((b, g, r), axis=2)
        bgr[:, :, 0] = b_n
        return bgr

    def render(self):

        img = np.copy(self.current_map_state)

        reward_map = img

        """ initialize heatmap """

        full_heatmap = self.heatmap_render_prep(reward_map)
        full_heatmap = cv2.resize(full_heatmap, (700, 700), interpolation=cv2.INTER_AREA)
        cv2.imshow("heatMap", full_heatmap)

        agent_views_list = []
        # =============================================================================
        #         for agent_indx, local_view in enumerate(self.agent_local_view_list):
        #
        #             temp = self.exploration_render_prep(local_view)
        #
        #             agent_views_list.append(temp)
        # =============================================================================

        for agent_indx, local_view in enumerate(self.local_heatmap_list):
            temp = self.heatmap_render_prep(local_view)

            agent_views_list.append(temp)

        rows = []
        for j in range(CONST.RENDER_ROWS):
            rows.append(np.hstack((agent_views_list[j * CONST.RENDER_COLUMNS: (j + 1) * CONST.RENDER_COLUMNS])))

        agent_views = np.vstack((rows))

        displayImg = cv2.resize(agent_views, (CONST.RENDER_COLUMNS * 200, CONST.RENDER_ROWS * 200),
                                interpolation=cv2.INTER_AREA)

        #        displayImg = cv2.resize(agent_views_list[0],(200,200),interpolation = cv2.INTER_AREA)
        cv2.imshow("Agent Views", displayImg)

        agent_views_list = []
        for agent_indx, minimap in enumerate(self.mini_map):
            temp = self.heatmap_render_prep(minimap)

            agent_views_list.append(temp)

        rows2 = []
        for j in range(CONST.RENDER_ROWS):
            rows2.append(np.hstack((agent_views_list[j * CONST.RENDER_COLUMNS: (j + 1) * CONST.RENDER_COLUMNS])))

        agent_views2 = np.vstack((rows2))

        minimapImg = cv2.resize(agent_views2, (CONST.RENDER_COLUMNS * 200, CONST.RENDER_ROWS * 200),
                                interpolation=cv2.INTER_AREA)

        #        displayImg = cv2.resize(agent_views_list[0],(200,200),interpolation = cv2.INTER_AREA)
        cv2.imshow("Minimap Views", minimapImg)

        #        mini_heatmap_img = self.heatmap_render_prep(self.mini_map[0])
        #
        #        mini_heatmap_img = cv2.resize(mini_heatmap_img,(350,350),interpolation = cv2.INTER_AREA)
        #
        #        cv2.imshow("Mini Heat Map", mini_heatmap_img)
        cv2.waitKey(50)

    def get_reward(self, current_map):

        # sum up reward on all free pixels
        actualR = np.where((current_map <= 0), current_map, 0)  # np.where(condition, x, y)满足条件(condition)，输出x，不满足输出y。

        curSumR = np.sum(actualR)

        return curSumR

    def get_reward_local(self, local_map_list, current_map):
        local_reward_list = []
        # sum up reward on all free pixels
        for local_map in local_map_list:
            actualR = np.where((local_map <= 0), local_map, 0)
            curSumR = np.sum(actualR)
            local_reward_list.append(curSumR)
        sharedR = np.where((current_map <= 0), current_map, 0)
        shared_reward = np.sum(sharedR)
        #        print(local_reward_list, shared_reward)
        return local_reward_list, shared_reward

    def get_local_heatmap_list(self, current_map, agent_g_pos_list):
        local_heatmap_list = []
        for g in agent_g_pos_list:
            r = int((CONST.LOCAL_SZ - 1) / 2)
            lx = int(max(0, g[1] - r))
            hx = int(min(CONST.MAP_SIZE, r + g[1] + 1))
            ly = int(max(0, g[0] - r))
            hy = int(min(CONST.MAP_SIZE, r + g[0] + 1))
            tempMask = np.zeros_like(current_map)
            tempMask[lx: hx, ly: hy] = 1

            local_view = np.ones((CONST.LOCAL_SZ, CONST.LOCAL_SZ)) * 150

            llx = int(lx - (g[1] - r))
            hhx = int(hx - g[1] + r)

            lly = int(ly - (g[0] - r))
            hhy = int(hy - g[0] + r)

            local_view[llx: hhx, lly: hhy] = current_map.T[lx: hx, ly: hy]
            local_heatmap_list.append(local_view.T)
        return local_heatmap_list

    def get_nodes_num(self,num):
        return obsMap.patrol_nodes[num]

    def get_neighbor_nodes_number(self,num):

        list = obsMap.get_neighbor_nodes_number(obsMap.patrol_nodes[num])
        idx = []
        for item in list:
            idx.append(obsMap.patrol_nodes.index(item))

        return idx

    def get_mini_map(self, current_map, ratio, agent_g_pos):
        num_windows = int(current_map.shape[0] * ratio)

        window_sz = int(1 / ratio)

        mini_obs = cv2.resize(self.obstacle_map, (num_windows, num_windows), interpolation=cv2.INTER_AREA)
        mini_obs = np.where(mini_obs > 0, 150, 0)

        decay_map = np.where(current_map < 0, current_map, 0)
        mini_decay = skimage.measure.block_reduce(decay_map, (window_sz, window_sz), np.min)

        mini_heatmap = np.where(mini_decay < 0, mini_decay, mini_obs)

        for gpos in agent_g_pos:
            mini_heatmap[int(gpos[0] * ratio), int(gpos[1] * ratio)] = 100

        agent_minimap_list = []
        for gpos in agent_g_pos:
            agent_minimap = np.copy(mini_heatmap)
            agent_minimap[int(gpos[0] * ratio), int(gpos[1] * ratio)] = 200
            agent_minimap_list.append(agent_minimap)
        return agent_minimap_list

    def save2Vid(self, episode, step):

        img = np.copy(self.current_map_state)

        reward_map = img

        """ initialize heatmap """

        full_heatmap = self.heatmap_render_prep(reward_map)
        full_heatmap = cv2.resize(full_heatmap, (700, 700), interpolation=cv2.INTER_AREA)
        display_string = "Episode: " + str(episode) + " Step: " + str(step)
        full_heatmap = cv2.putText(full_heatmap, display_string, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                   (255, 255, 255), 2, cv2.LINE_AA)
        self.out.write(full_heatmap.astype('uint8'))


