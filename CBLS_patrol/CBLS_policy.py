import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import os
from Env.env import obsMap

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import math
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import time

from Env.env import Env
from Env.env import CONST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val[0] == value[0] and val[1] == value[1]:
            return key


class ConcurrentBayesianLearning():

    def __init__(self, env):
        self.theta_adj = np.ones_like(env.adj, dtype=np.float64)
        self.graph_adj = env.adj

    def reset(self, env):
        self.theta_adj = np.ones_like(env.adj, dtype=np.float64)
        self.graph_adj = env.adj

    def choose(self, state, nodes_last_list, visits, decision_flag, nodes_last_goal=None):

        nodes = []
        norm_factor = np.sum(self.theta_adj) / 2
        visitflag = [False for i in range(len(obsMap.patrol_nodes))]

        for i in range(CONST.NUM_AGENTS):
            if not decision_flag[0][i]:
                nodes.append(nodes_last_goal[i])
                k = obsMap.patrol_nodes.index(get_key(nodes_last_goal[i], obsMap.nodes))
                visitflag[k] = True

        for i in range(CONST.NUM_AGENTS):
            if not decision_flag[0][i]:
                continue

            node_last_visit = []
            node_last_visit.append(
                obsMap.get_key(nodes_last_list[i], obsMap.nodes))
            a = node_last_visit[0]
            nb_list = obsMap.get_neighbor_nodes_number(a)
            current_node = obsMap.patrol_nodes.index(a)

            selected = -1
            judge_time = 0
            sum_idleness_nb = 0
            for j_j in nb_list:
                i = obsMap.patrol_nodes.index(j_j)
                sum_idleness_nb += state[i][2]
            p_move = 0
            Entropy_move = 0
            for j in nb_list:

                i = obsMap.patrol_nodes.index(j)

                p_move_i = state[i][2] / (sum_idleness_nb + 0.01)
                p_move_edge = self.theta_adj[current_node, i]

                p_move_edge = p_move_edge / np.abs(norm_factor + 0.01)
                if p_move_edge > 20:
                    p_move_edge = 20
                p_move_i = p_move_i * (math.exp(p_move_edge) - 0.67)
                Entropy_move -= p_move_i * math.log2(p_move_i + 1)

                if p_move <= p_move_i and not visitflag[i]:
                    p_move = p_move_i
                    selected = j
                    visitflag[i] = True

            if selected == -1:
                selected = nb_list[0]

            nodes.append(obsMap.nodes[selected])

            ent_norm = len(nb_list)
            Entropy_move = Entropy_move / math.log2(ent_norm + 0.01)
            for j in nb_list:
                i = obsMap.patrol_nodes.index(j)
                S_i_currentnode = self.compute_S(visits, state, i, current_node)
                gamma = S_i_currentnode * (1 - Entropy_move)
                self.theta_adj[i, current_node] += gamma
                self.theta_adj[current_node, i] += gamma

        log_prob_choose = [torch.tensor(0, device=device) for i in range(CONST.NUM_AGENTS)]
        return nodes, log_prob_choose

    def compute_S(self, visits, state, i, current_node):
        S = 0
        neighbors = self.graph_adj[current_node]
        beta = 0
        argmax_i = -1
        argmin_i = -1
        max_zeta = -1
        min_zeta = 1000
        for i_ in range(len(neighbors)):
            if neighbors[i_] != 0:
                beta += 1
                deg_i = self.graph_adj[i_]
                deg_i = np.nonzero(deg_i)
                deg_i = len(deg_i[0])
                zeta = visits[i_] / deg_i

                if zeta >= max_zeta:
                    if zeta > max_zeta:
                        max_zeta = zeta
                        argmax_i = i_
                    if zeta == max_zeta:
                        if state[argmax_i][2] > state[i_][2]:
                            argmax_i = i_

                if zeta <= min_zeta:
                    if zeta < min_zeta:
                        min_zeta = zeta
                        argmin_i = i_
                    if zeta == min_zeta:
                        if state[argmin_i][2] < state[i_][2]:
                            argmin_i = i_

        if beta > 1 and argmax_i == i:
            S = -1
        elif beta > 1 and argmin_i == i:
            S = 1
        else:
            S = 0
        return S

