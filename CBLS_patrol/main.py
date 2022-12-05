import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from Env.env import obsMap

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import time
from Env.env import Env
from CBLS_policy import ConcurrentBayesianLearning
from CBLS_policy import get_key
from Env.env import CONST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    env = Env()
    CBLS = ConcurrentBayesianLearning(env)
    idleness = []
    idleness_greedy = []
    idleness_rand = []
    idleness_bayes = []
    batch_step = 0
    Global_idle = []
    Global_idle_greedy = []
    GLobal_idle_bayes = []
    Global_ent = []
    Global_ent_greedy = []
    Global_ent_bayes = []
    all_ep_r = []

    for episode in range(800):

        state, nodes_last_visit, last_vertex, neighbor_nodes, flag = env.reset()
        CBLS.reset(env)
        SDI = []
        SDI_greedy = []
        SDI_bayes = []
        nodes_last_goal = [nodes_last_visit[i] for i in range(CONST.NUM_AGENTS)]
        episode_reward = 0
        t0 = time.time()

        change_reward = 0
        change_step = 0
        decision_index = -torch.ones((1, CONST.NUM_AGENTS)).to(device)

        for t in range(1000):

            if (episode + 1) % 100 == 0:
                env.render()

            decision_flag = torch.ones((1, CONST.NUM_AGENTS), dtype=bool).to(device)
            greedynodes = []

            # state (batch_size=1,graph_size,3)
            # nodes_last_list : ( batch_size=1,num_agent)
            # decision_index  : (batch_size=1, num_agent)
            # decision_flag  : (batch_size=1,  num_agent)
            # choose_node(self, state, nodes_last_list,decision_index,decision_flag)

            for i in range(CONST.NUM_AGENTS):
                decision_flag[0][i] = flag[i]

            if t == 0:
                nodes, log_prob = CBLS.choose(state, nodes_last_visit, Env.visits, decision_flag)

            elif any(decision_flag[0]):

                nodes, log_prob = CBLS.choose(state, nodes_last_visit, Env.visits, decision_flag, nodes)
                change_reward = 0


            # choose node: list: num_agent,2 log_prob_choose: list: num_agent
            # state:(1,graph_size,3)  nodes_last_visit:(1,num_agents)

            else:
                nodes = []
                for i in range(CONST.NUM_AGENTS):
                    nodes.append(obsMap.nodes[int(decision_index[0][i])])

            a = time.time()
            # next_state

            # update decision_index:

            decision_index = [get_key(nodes[i], obsMap.nodes) for i in range(CONST.NUM_AGENTS)]
            decision_index = np.array(decision_index)
            decision_index = torch.tensor(decision_index, device=device).unsqueeze(0)  # (1,num_agents)
            next_state, agent_pos_list, current_map_state, nb_nodes, local_heatmap_list, mini_map, \
            shared_reward, flag, last_vertex_list, n_last_visit, visit_flag, done = env.step(
                nodes, nodes_last_visit, last_vertex, state)

            if episode == 0:
                idleness.append(env.avg_i)
                SDI.append(env.interval)
            elif episode == 1:
                idleness_greedy.append(env.avg_i)
                SDI_greedy.append(env.interval)
            else:
                idleness_bayes.append(env.avg_i)
                SDI_bayes.append(env.interval)
            last_vertex = [last_vertex_list[i] for i in range(CONST.NUM_AGENTS)]
            nodes_last_goal = [nodes[i] for i in range(CONST.NUM_AGENTS)]
            change_reward += shared_reward  # change_reward
            if any(decision_flag[0]):
                change_step += 1

            state = next_state
            nodes_last_visit = n_last_visit
            episode_reward += shared_reward

        if episode == 0:
            all_ep_r.append(episode_reward)
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + episode_reward * 0.1)

        if (episode + 1) % 10 == 0:
            print(all_ep_r)
        print(
            'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f} | Idleness : {:.4f} | Global Idleness : {:.4f}'.format(
                episode, 800, episode_reward,
                time.time() - t0, env.avg_idleness, env.avg_i_sum / CONST.LEN_EPISODE
            )
        )
        plt.cla()
