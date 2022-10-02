import math

import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from matplotlib.animation import FuncAnimation
import time

from env import obsMap
from d2l import torch as d2l
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import sys
sys.path.append("..")
from entropy_compute import get_entropy
from entropy_estimators import continuous
import time


from ma_encoder import GraphEncoder
from ma_decoder import AttentionModel
from env import Env
from env import CONST



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed =21
np.random.seed(seed)
torch.manual_seed(seed)


class Policy(nn.Module):
    def __init__(self, env,hidden_dim=32,action_dim=25):
        super(Policy, self).__init__()
        self.encoder = GraphEncoder(env).to(device)
        self.decoder = AttentionModel(env).to(device)
        self.fc1 = nn.Linear(5, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.ln = nn.LayerNorm([32]).to(device)
        self.prob = nn.Linear(hidden_dim, action_dim).to(device)
        #self.decoder2=nn.Sequential(self.fc1,nn.ReLU(),self.ln(),self.fc2,nn.ReLU(),self.ln,self.prob)
        self.adj=env.adj
        self.select_type="sampling"
        self.value = nn.Linear(25*hidden_dim, 1).to(device)


    def forward(self, state, node_last_visit,decision_index,decision_flag,shift_action=None):

        state=torch.tensor(state).to(device)
        state=state.unsqueeze(1)
        state=torch.repeat_interleave(state, CONST.NUM_AGENTS, dim=1)  # (batch num_agent ,graph_size,3)

        batch_size=len(node_last_visit)
        graph_size=state.shape[-2]
        mask = np.zeros((batch_size, CONST.NUM_AGENTS, graph_size))
        zuobiao=torch.zeros((batch_size,CONST.NUM_AGENTS,2)).to(device)
        idx = torch.zeros(batch_size, CONST.NUM_AGENTS).to(device)
        for i in range(batch_size):
            for j in range(CONST.NUM_AGENTS):
                zuobiao[i][j]=torch.tensor(obsMap.nodes[node_last_visit[i][j]]).to(device)
                idx[i][j] = torch.tensor(obsMap.patrol_nodes.index(node_last_visit[i][j])).to(device)
                z = idx[i][j]
                mask[i][j] = self.adj[int(z.cpu().detach().numpy())]
        mask = torch.tensor(mask, device=device, requires_grad=False)

        zuobiao=zuobiao.unsqueeze(2)

        zuobiao=torch.repeat_interleave(zuobiao,graph_size,dim=2) # graph_size=1
        state=torch.cat((state,zuobiao),-1)

        node_embedding, state_value = self.encoder(state, node_last_visit)
        #node_embedding, state_value = self.encoder(zuobiao, node_last_visit) # (batch_size,num_agents,grpah_size,hidden_dim)
        n_embedding, s_value = self.encoder(state, node_last_visit)
        #node_embedding=self.fc1(state)  #encoder2
        X=node_embedding
        Y = X.contiguous().reshape(X.shape[0], X.shape[1], -1)
        #state_value = self.value(Y)

        selected_total, log_prob_total,shift_action_total = self.decoder(node_embedding,
                                                      node_last_visit,decision_index,decision_flag,shift_action)



        return selected_total, log_prob_total, state_value,shift_action_total

    def _select_node(self, probs):
        select=[]
        for i in range(len(probs)):
           probs_i = probs[i].squeeze()
           if self.select_type == 'greedy':
              _, selected = probs_i.max(1)

           elif self.select_type == 'sampling':

                  selected = probs_i.multinomial(1)
                  selected=selected.squeeze()

           select.append(selected)
        select=torch.stack(select).to(device)
           #select=select.reshape(probs.shape[0],probs.shape[1])
        return select



class ConcurrentBayesianLearning():

    def __init__(self,env):
        self.theta_adj=np.ones_like(env.adj,dtype=np.float64)
        self.graph_adj=env.adj
    def reset(self,env):
        self.theta_adj = np.ones_like(env.adj, dtype=np.float64)
        self.graph_adj = env.adj
    def choose(self, state, nodes_last_list,visits):

        nodes = []
        norm_factor = np.sum(self.theta_adj)/2
        visitflag = [False for i in range(len(obsMap.patrol_nodes))]
        for i in range(CONST.NUM_AGENTS):
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

                p_move_i = state[i][2] / (sum_idleness_nb+0.01)
                p_move_edge = self.theta_adj[current_node, i]

                p_move_edge = p_move_edge / np.abs(norm_factor+0.01)
                p_move_i = p_move_i  *(math.exp(p_move_edge)-0.8)
                Entropy_move -= p_move_i * math.log2(p_move_i+0.5)

                if p_move <=p_move_i and not visitflag[j]:
                    p_move = p_move_i
                    selected = j
                    visitflag[j] = True

            if selected == -1:
                r=np.random.randint(0,len(nb_list))
                selected = nb_list[r]

            nodes.append(obsMap.nodes[selected])

            ent_norm = len(nb_list)
            Entropy_move = Entropy_move / math.log2(ent_norm+0.01)
            for j in nb_list:
                i = obsMap.patrol_nodes.index(j)
                S_i_currentnode=self.compute_S(visits,state,i,current_node)
                gamma=S_i_currentnode*(1-Entropy_move)
                self.theta_adj[i,current_node]+=gamma
                self.theta_adj[current_node,i]+=gamma

        log_prob_choose = [torch.tensor(0, device=device) for i in range(CONST.NUM_AGENTS)]
        return nodes,log_prob_choose

    def compute_S(self,visits,state,i,current_node):
        S=0
        neighbors=self.graph_adj[current_node]
        beta=0
        argmax_i=-1
        argmin_i=-1
        max_zeta=-1
        min_zeta=1000
        for i_ in range(len(neighbors)):
            if neighbors[i_]!=0:
                beta+=1
                deg_i=self.graph_adj[i_]
                deg_i=np.nonzero(deg_i)
                deg_i=len(deg_i[0])
                zeta=visits[i_]/deg_i

                if zeta>=max_zeta:
                    if zeta>max_zeta:
                      max_zeta=zeta
                      argmax_i=i_
                    if zeta==max_zeta:
                        if state[argmax_i][2]>state[i_][2]:
                            argmax_i=i_

                if zeta<=min_zeta:
                    if zeta<min_zeta:
                      min_zeta=zeta
                      argmin_i=i_
                    if zeta==min_zeta:
                        if state[argmin_i][2]<state[i_][2]:
                            argmin_i=i_

        if beta>1 and argmax_i==i :
            S=-1
        elif beta>1 and argmin_i==i:
            S=1
        else:
            S=0
        return S

class PPO():
    def __init__(self, env):

        self.gamma = 0.99
        self.policy_lr = 0.000001
        self.train_episodes = 3
        # 看一下网络参数的大小
        self.test_episodes = 100
        self.len_episode = CONST.LEN_EPISODE
        self.policy_update_steps = 5
        self.batch_size = 20
        # 每10次更新网络
        self.METHOD = [
            dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
            dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
        ][1]
        self.eps = 1e-8
        self.ent_coef = 0
        self.vf_coef = 0.8
        self.env = env
        '''
        #self.critic=GraphEncoder(self.env).to(device)
        #self.critic.train()

        self.actor=AttentionModel(self.env)
        self.actor_old=AttentionModel(self.env)
        self.actor.train()
        self.actor_old.eval()
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=self.actor_lr)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=self.critic_lr)
        '''
        self.policy = Policy(self.env)
       # self.policy=Policy2(self.env)
        self.policy.train()
        # self.policy_old=Policy(self.env).to(device)
        # self.policy_old.eval()
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr,eps=1e-5)
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.policy_opt,mode='max',factor=0.5,patience=75)
        self.loss = nn.MSELoss()

        self.state_buffer, self.action_buffer = [], []
        self.shift_action_buffer=[]
        self.decision_index_buffer, self.decision_flag_buffer = [], []  # 相当于状态


        self.reward_buffer = []
        self.nodes_last_visit_buffer = []

        self.log_prob_buffer = []
        self.G_buffer = []

        self.nodes_visit_buffer = []



    def store_transition(self, state, decision_index,decision_flag,action, reward, nodes_last_visit,
                         log_prob, nodes_visit):

        self.state_buffer.append(state)
        self.decision_index_buffer.append(decision_index)
        self.decision_flag_buffer.append(decision_flag)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        #self.shift_action_buffer.append(shift_action)
        self.nodes_last_visit_buffer.append(nodes_last_visit)

        self.log_prob_buffer.append(log_prob)
        self.G_buffer.append(env.avg_idleness)

        self.nodes_visit_buffer.append(nodes_visit)



    # state (batch_size=1,num_agent,graph_size,3)
    # nodes_last_list : ( batch_size=1,num_agent) np.array
    # decision_index  : (batch_size=1, num_agent)
    # decision_flag  : (batch_size=1,  num_agent)
    def choose_node(self, state, nodes_last_list,decision_index,decision_flag):

        node_last_visit = []
        for i in range(CONST.NUM_AGENTS):
            node_last_visit.append(
                obsMap.get_key(nodes_last_list[i], obsMap.nodes))
        state=np.array(state)

        state=torch.tensor(state).to(device).unsqueeze(0) # (1,graph_size,3)

        node_last_visit=np.array(node_last_visit)

        node_last_visit=node_last_visit[None,:] #(1,num_agent)

        selected, log_prob, s_v,shift_action = self.policy(state, node_last_visit,decision_index,decision_flag)
        # (batch_size=1,num_agents) # (batch_size=1,num_agent,graph_size)
        selected=[item for item in selected[0]]
        writer.add_scalar("state_value", s_v.mean(), episode * self.len_episode + t)
        #writer.add_scalar("current node", np.array(node_last_visit), episode * self.len_episode + t)
        log_prob_choose=[]
        for i in range(CONST.NUM_AGENTS):
            log_prob_choose.append(log_prob[0][i][selected[i].cpu().detach().numpy()])
        nodes=[]
        for i in range(CONST.NUM_AGENTS):
            cc=selected[i].cpu().detach().numpy()
            cc=np.array(cc,dtype=int)
            dd=obsMap.patrol_nodes[cc]
            nodes.append(obsMap.nodes[dd])
        return nodes, log_prob_choose,shift_action
        # return : choose的node : list: num_agent,2 , log_prob_choose : list : num_agent



    # choose的node: list: num_agent,2 log_prob_choose: list: num_agent
    # state:(1,graph_size,3)  nodes_last_visit:(1,num_agents)
    def greedy_choose_node(self, state, nodes_last_list):


        nodes=[]

        visitflag=[False for i in range(len(obsMap.patrol_nodes))]
        for i in range(CONST.NUM_AGENTS):
            node_last_visit = []
            node_last_visit.append(
                obsMap.get_key(nodes_last_list[i], obsMap.nodes))
            a = node_last_visit[0]
            nb_list = obsMap.get_neighbor_nodes_number(a)

            selected = -1
            judge_time = 0
            for j in nb_list:
               i = obsMap.patrol_nodes.index(j)
               if judge_time <= state[i][2] and not visitflag[j]:
                  judge_time = state[i][2]
                  selected = j
                  visitflag[j]=True
            if selected==-1:
                selected=nb_list[0]

            nodes.append(obsMap.nodes[selected])
        log_prob_choose=[torch.tensor(0,device=device) for i in range(CONST.NUM_AGENTS)]
        return nodes,log_prob_choose

    def randchoose(self,state,nodes_last_list):
        nodes = []

        visitflag = [False for i in range(len(obsMap.patrol_nodes))]
        for i in range(CONST.NUM_AGENTS):
            node_last_visit = []
            node_last_visit.append(
                obsMap.get_key(nodes_last_list[i], obsMap.nodes))
            a = node_last_visit[0]
            nb_list = obsMap.get_neighbor_nodes_number(a)

            selected = 0
            judge_time = 0
            i=np.random.randint(0,len(nb_list))
            selected=nb_list[i]

            nodes.append(obsMap.nodes[selected])
        log_prob_choose = [torch.tensor(0, device=device) for i in range(CONST.NUM_AGENTS)]
        return nodes, log_prob_choose



    '''
     '''


    def update_greedy(self):
        self.state_buffer.clear()
        self.decision_index_buffer.clear()
        self.decision_flag_buffer.clear()
        self.action_buffer.clear()
        self.shift_action_buffer.clear()
        self.reward_buffer.clear()
        self.log_prob_buffer.clear()
        self.G_buffer.clear()

        self.nodes_last_visit_buffer.clear()
        self.nodes_visit_buffer.clear()

    def gae(self):

        state_next,node_next,shift_action_next=self.state_buffer[-1],self.nodes_last_visit_buffer[-1],\
                                               self.shift_action_buffer[-1]
        decision_index_next,decision_flag_next=self.decision_index_buffer[-1],self.decision_flag_buffer[-1]
        st_buffer=[]
        n_last_visit_buffer=[]
        d_index_buffer=[]
        d_flag_buffer=[]
        shift_a_buffer=[]
        for i in range(len(self.state_buffer)):
            if i <len(self.state_buffer)-1:
                st_buffer.append(self.state_buffer[i])
                n_last_visit_buffer.append(self.nodes_last_visit_buffer[i])
                d_index_buffer.append(self.decision_index_buffer[i])
                d_flag_buffer.append(self.decision_flag_buffer[i])
                shift_a_buffer.append(self.shift_action_buffer[i])
        s = np.array(st_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.reward_buffer, np.float32)
        d_index_buffer=torch.stack(d_index_buffer)
        d_flag_buffer=torch.stack(d_flag_buffer)
        shift_a_=torch.stack(shift_a_buffer)
        n = n_last_visit_buffer

        _,_,next_value,_=self.policy(np.array(state_next)[None,:,:],
                                   [node_next],decision_index_next.unsqueeze(0),
                                   decision_flag_next.unsqueeze(0),shift_action_next)
        G__estimate_list=[]
        next_value=next_value.mean().cpu().detach().numpy()
        for item in self.reward_buffer[::-1]:
            G_estimate_t = self.gamma * next_value + item
            next_value=G_estimate_t
            G__estimate_list.append(G_estimate_t)
        G__estimate_list.reverse()
        G_estimate_Tensor=torch.Tensor(G__estimate_list).to(device)
        _,_,value,_=self.policy(s,n,d_index_buffer,d_flag_buffer,shift_a_.squeeze(1))
        advantage=G_estimate_Tensor.squeeze()-value.mean(axis=1).squeeze()
        return advantage,G_estimate_Tensor.squeeze()



    def saveModel(self, filePath, per_save=False, episode=0):

        if per_save == False:
            torch.save(self.policy.state_dict(), f"{filePath}/multipolicy_.pt")

        else:
            torch.save(self.policy.state_dict(), f"{filePath}/multipolicy_{episode}.pt")

    def loadModel(self, filePath, cpu=1):

        if cpu == 1:
            self.policy.load_state_dict(torch.load(f"{filePath}/strip_2agent_b.pt", map_location=torch.device('cpu')))
#
       # else:   STRIP MAP multipolicy_4agent20_204  GRID MAP 0agent50_251

        #   self.policy.load_state_dict(torch.load(f"{filePath}/multipolicy_4agent20_201.pt"))

        self.policy.eval()


# main


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val[0] == value[0] and val[1] == value[1]:
            return key








env = Env()
ppo = PPO(env)

CBLS=ConcurrentBayesianLearning(env)
ppo.loadModel("../checkpoints")
run_name = f"final-test{seed}_{int(time.time())}"
writer = SummaryWriter(log_dir=f"../runs/{run_name}")
model1 = GraphEncoder(env).to(device)
model2 = AttentionModel(env).to(device)
init_x = torch.zeros((1, 20, 3), device="cpu")

init_x = torch.Tensor(init_x)
# init_y,_=model1(init_x)
# writer.add_graph(model1,init_x)
# writer.add_graph(model2,(init_y,[0]))


all_ep_r = []
animator = d2l.Animator(xlabel='episode', ylabel='episode_reward',
                        legend=['train'], xlim=[10, ppo.train_episodes])

batch_step = 0
#idleness=[]
#idleness_greedy=[]
idleness=[]
idleness_greedy=[]

idleness_bayes=[]


Global_idle=[]
Global_idle_greedy=[]
GLobal_idle_bayes=[]
Global_ent=[]
Global_ent_greedy=[]
Global_ent_bayes=[]
for episode in range(ppo.train_episodes):
    # state=env.reset()

    state, nodes_last_visit, last_vertex, neighbor_nodes, flag = env.reset()

    CBLS.reset(env)
    SDI = []
    SDI_greedy = []
    SDI_bayes = []
    nodes_last_goal = [nodes_last_visit[i] for i in range(CONST.NUM_AGENTS)]
    episode_reward = 0
    t0 = time.time()
    ppo.nodes_last_visit_buffer.clear()
    ppo.state_buffer.clear()
    ppo.decision_flag_buffer.clear()
    ppo.decision_index_buffer.clear()
    ppo.reward_buffer.clear()
    ppo.action_buffer.clear()
    ppo.shift_action_buffer.clear()

    change_reward = 0
    change_step = 0
    decision_index = -torch.ones((1, CONST.NUM_AGENTS)).to(device)


    for t in range(ppo.len_episode):

        env.render()
        decision_flag=torch.ones((1,CONST.NUM_AGENTS),dtype=bool).to(device)
        greedynodes = []

        #            if i == 0:
        #                actions.append(getKeyPress())
        #            else:
        #                actions.append(0)
        # state (batch_size=1,graph_size,3)
        # nodes_last_list : ( batch_size=1,num_agent)
        # decision_index  : (batch_size=1, num_agent)
        # decision_flag  : (batch_size=1,  num_agent)
        # choose_node(self, state, nodes_last_list,decision_index,decision_flag)

        for i in range(CONST.NUM_AGENTS):

            decision_flag[0][i]=flag[i]
        if t==0:
            ppo.state_buffer.append(state)
            ppo.decision_index_buffer.append(decision_index.squeeze())
            ppo.decision_flag_buffer.append(decision_flag.squeeze())

            ppo.nodes_last_visit_buffer.append([get_key(torch.Tensor(nodes_last_visit[i]).squeeze() ,
                                         obsMap.nodes)for i in range(CONST.NUM_AGENTS)])
            if episode==0:
              nodes, log_prob,shift_action = ppo.choose_node(state, nodes_last_visit, decision_index, decision_flag)
            elif episode==1:
              shift_action = torch.zeros((1, CONST.NUM_AGENTS, CONST.NUM_AGENTS, 2)).to(device)
              ppo.shift_action_buffer.append(shift_action)
              nodes, log_prob = ppo.greedy_choose_node(state, nodes_last_visit)

            else:
                shift_action = torch.zeros((1, CONST.NUM_AGENTS, CONST.NUM_AGENTS, 2)).to(device)
                ppo.shift_action_buffer.append(shift_action)
                nodes,log_prob=CBLS.choose(state,nodes_last_visit,Env.visits)

        elif any(decision_flag[0]):
            ppo.store_transition(state, decision_index.squeeze(),decision_flag.squeeze(),nodes, change_reward,
                                 [get_key(torch.Tensor(nodes_last_visit[i]).squeeze() ,
                                         obsMap.nodes)for i in range(CONST.NUM_AGENTS)], log_prob,
                                 [get_key(torch.Tensor(n_last_visit[i]).squeeze(), obsMap.nodes)
                                  for i in range(CONST.NUM_AGENTS)])

            if episode==0:
               nodes,log_prob,shift_action=ppo.choose_node(state,nodes_last_visit,decision_index,decision_flag)
            elif episode==1:
               shift_action=torch.zeros((1,CONST.NUM_AGENTS,CONST.NUM_AGENTS,2)).to(device)
               ppo.shift_action_buffer.append(shift_action)
               nodes,log_prob=ppo.greedy_choose_node(state,nodes_last_visit)
               c=1

            else:
                shift_action = torch.zeros((1, CONST.NUM_AGENTS, CONST.NUM_AGENTS, 2)).to(device)
                ppo.shift_action_buffer.append(shift_action)
                nodes, log_prob = CBLS.choose(state, nodes_last_visit, Env.visits)

            change_reward=0
        # choose的node: list: num_agent,2 log_prob_choose: list: num_agent
        # state:(1,graph_size,3)  nodes_last_visit:(1,num_agents)

        else:
            nodes = []
            for i in range(CONST.NUM_AGENTS):
                nodes.append(obsMap.nodes[int(decision_index[0][i])])






        a = time.time()

        # greedy_env_step:
        '''if m:
            env.store_last()
            _, _, _,_, _, _, \
             _, _, _, _, _, _ = env.step(
            greedynodes, nodes_last_visit, last_vertex, state)
            greedy_G=env.avg_idleness
            env.return_last()'''

        # env.store_last()
        # env.return_last()
        # greedy_G=0
        # env.step：


        decision_index=[ get_key(nodes[i],obsMap.nodes) for i in range(CONST.NUM_AGENTS)]
        decision_index=np.array(decision_index)
        decision_index=torch.tensor(decision_index,device=device).unsqueeze(0) # (1,num_agents)
        next_state, agent_pos_list, current_map_state, nb_nodes, local_heatmap_list, mini_map, \
        shared_reward, flag, last_vertex_list, n_last_visit, visit_flag, done = env.step(
            nodes, nodes_last_visit, last_vertex, state)
        state = np.array(state)
        if 0<=episode<20:
        #   idleness.append(env.avg_i)

           SDI.append(env.interval)
        elif 20<=episode<40:
            #idleness_greedy.append(env.avg_i)
            SDI_greedy.append(env.interval)
        else:
            SDI_bayes.append(env.interval)

        #else:
        #    idleness_bayes.append(env.avg_i)
        #    SDI_bayes.append(env.interval)
        last_vertex = [last_vertex_list[i] for i in range(CONST.NUM_AGENTS)]
        nodes_last_goal = [nodes[i] for i in range(CONST.NUM_AGENTS)]
        change_reward += shared_reward

        if any(decision_flag[0]):


            change_step += 1

        state = next_state
        nodes_last_visit = n_last_visit

        episode_reward += shared_reward


        '''if (change_step + 1) % ppo.batch_size == 0 and m and len(ppo.state_buffer) \
                or all(visit_flag) and len(ppo.state_buffer):
            ppo.finish_path(next_state,nodes_last_visit, done)
            ppo.update()
        if all(visit_flag):
            print(t)
            break
        '''
        #x.append(t)
        #y.append(env.avg_i)

        #line.set_xdata(x)
        #line.set_ydata(y)
        #plt.pause(1e-17)
        if episode==0:
            idleness.append(env.avg_i)
        if episode==1:
            idleness_greedy.append(env.avg_i)
        if episode==2:
            idleness_bayes.append(env.avg_i)
        if change_step % ppo.batch_size == 0 and any(decision_flag[0]) and len(ppo.reward_buffer) or t == ppo.len_episode :
            cc = 1
            ppo.update_greedy()
            batch_step += 1

    if episode == 0:
        all_ep_r.append(episode_reward)
    else:
        all_ep_r.append(all_ep_r[-1] * 0.9 + episode_reward * 0.1)
    writer.add_scalar("episode_reward", episode_reward, episode)

    if (episode + 1) % 10 == 0:
        animator.add(episode + 1, episode_reward)
        print(all_ep_r)
    print(
        'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f} | Idleness : {:.4f} | Global Idleness : {:.4f}'.format(
            episode, ppo.train_episodes, episode_reward,
            time.time() - t0, env.avg_idleness,env.avg_i_sum/CONST.LEN_EPISODE
        )
    )

    if episode==2:
        x = []
        y1 = []
        y2=[]
        y3=[]
        axes = plt.gca()
        axes.set_xlim(0, 1000)
        axes.set_ylim(0, 100)
        line1, = axes.plot(x, y1, 'm-',label='MARL')
        line2, =axes.plot(x,y2,'c-',label='Greedy')
        line3, =axes.plot(x,y3,'r-',label='CBLS')
        plt.legend()
        plt.grid()
        plt.xlabel('Env Timestep', fontsize=20)
        plt.ylabel('Env IGI(t)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title('Real-time Idleness Curve', fontsize=20)

        for step in range(ppo.len_episode):
            x.append(step)
            y1.append(idleness[step]*0.8)
            y2.append(idleness_greedy[step]*0.8)
            y3.append(idleness_bayes[step]*0.8)
            line1.set_xdata(x)
            line1.set_ydata(y1)
            line2.set_xdata(x)
            line2.set_ydata(y2)
            line3.set_xdata(x)
            line3.set_ydata(y3)
            plt.pause(1e-17)

        output=open('idleness','w',encoding='gbk')
        idleness=np.array(idleness)
        idleness_greedy=np.array(idleness_greedy)
        idleness_bayes=np.array(idleness_bayes)
        idle=[idleness,idleness_greedy,idleness_bayes]
        for i in range(len(idle)):
            for j in range(len(idle[i])):
                output.write(str(idle[i][j]))
                output.write('\t')
            output.write('\n')




    # x.append(t)
    # y.append(env.avg_i)

    # line.set_xdata(x)
    # line.set_ydata(y)
    # plt.pause(1e-17)


    if 0<=episode<20:
        Global_idle.append(env.avg_i_sum)
        sdi = []
        SDI = np.array(SDI).squeeze()
        for ii in range(len(env.patrol_points[0])):
            temp = SDI[:, ii]
            interval = temp[np.nonzero(temp)]
            l = len(interval)
            interval = interval.reshape(l, 1)
            # print(interval)
            # kozachenko = continuous.get_h(interval, k=5)
            kozachenko = get_entropy(interval)
            sdi.append(kozachenko)
        Global_ent.append(np.sum(np.array(sdi)))
        print(f"{episode}_entropy:{np.sum(np.array(sdi))}")
    if episode==19:
        Global_idle=np.array(Global_idle)

    if 20<=episode<40:

        Global_idle_greedy.append(env.avg_i_sum)
        sdi_greedy = []
        SDI_greedy = np.array(SDI_greedy).squeeze()
        for ii in range(len(env.patrol_points[0])):
            temp = SDI_greedy[:, ii]
            interval = temp[np.nonzero(temp)]
            l = len(interval)
            interval = interval.reshape(l, 1)
            # print(interval)
            # kozachenko = continuous.get_h(interval, k=5)
            kozachenko = get_entropy(interval)
            sdi_greedy.append(kozachenko)
        Global_ent_greedy.append(np.sum(np.array(sdi_greedy)))
        print(f"{episode}_entropy:{np.sum(np.array(sdi_greedy))}")
    if episode == 39:
        Global_idle_greedy = np.array(Global_idle_greedy)

    if 40<=episode<60:

        GLobal_idle_bayes.append(env.avg_i_sum)

        sdi_bayes = []
        SDI_bayes = np.array(SDI_bayes).squeeze()
        for ii in range(len(env.patrol_points[0])):
            temp = SDI_bayes[:, ii]
            interval = temp[np.nonzero(temp)]
            l = len(interval)
            interval = interval.reshape(l, 1)
            # print(interval)
            # kozachenko = continuous.get_h(interval, k=5)
            kozachenko = get_entropy(interval)
            sdi_bayes.append(kozachenko)
        Global_ent_bayes.append(np.sum(np.array(sdi_bayes)))
        print(f"{episode}_entropy:{np.sum(np.array(sdi_bayes))}")

    #ppo.saveModel("checkpoints")
    #print("Store Successfully !")
    if episode % 50 == 0:
        ppo.saveModel("../checkpoints",True)
    # plt.show()

Global_idle=np.array(Global_idle)/CONST.LEN_EPISODE
Global_idle_greedy=np.array(Global_idle_greedy)/CONST.LEN_EPISODE
GLobal_idle_bayes=np.array(GLobal_idle_bayes)/CONST.LEN_EPISODE
Global_ent=np.array(Global_ent)
Global_ent_greedy=np.array(Global_ent_greedy)
Global_ent_bayes=np.array(Global_ent_bayes)

data = [Global_idle, Global_idle_greedy, GLobal_idle_bayes]

fig = plt.figure(figsize=(10, 7))

# Creating plot
plt.boxplot(data)
plt.xlabel("algorithm")
plt.ylabel("Average Global Idleness")
plt.xticks(range(1,4),('RL','greedy','CBLS'))
# show plot
plt.show()

data = [Global_ent, Global_ent_greedy, Global_ent_bayes]

fig = plt.figure(figsize=(10, 7))

# Creating plot
plt.boxplot(data)
plt.xlabel("algorithm")
plt.ylabel("Global visit time entropy")
plt.xticks(range(1,4),('RL','greedy','CBLS'))
# show plot
plt.show()

print(Global_idle)
print(Global_idle_greedy)
print(GLobal_idle_bayes)
print(np.mean(Global_idle))
print(np.mean(Global_idle_greedy))
print(np.mean(GLobal_idle_bayes))

tt=np.arange(0,ppo.len_episode,1)
idleness=np.array(idleness)
idleness_greedy=np.array(idleness_greedy)

idleness_bayes=np.array(idleness_bayes)
#plt.plot(tt,idleness)
#plt.plot(tt,idleness_greedy)
#plt.plot(tt,idleness_rand)
#plt.plot(tt,idleness_bayes)

#plt.show()




sdi_grd=[]
sdi_bys=[]

  #(T,graph_size)
SDI_greedy=np.array(SDI_greedy).squeeze()
SDI_bayes=np.array(SDI_bayes).squeeze()
for ii in range(len(env.patrol_points[0])):
    temp=SDI_bayes[:,ii]
    interval=temp[np.nonzero(temp)]
    l=len(interval)
    interval=interval.reshape(l,1)
    #print(interval)
    #kozachenko = continuous.get_h(interval, k=5)
    kozachenko=get_entropy(interval)
    sdi_bys.append(kozachenko)
print("bayes entropy")
print(np.array(sdi_bys)/len(env.patrol_points[0]))
print(np.sum(np.array(sdi_bys))/len(env.patrol_points[0]))
for ii in range(len(env.patrol_points[0])):
    temp=SDI[ii]
    interval=temp[np.nonzero(temp)]
    #print(interval)
    kozachenko = continuous.get_h(interval, k=7)
    sdi.append(kozachenko)
print('\n')
for ii in range(len(env.patrol_points[0])):
    temp=SDI_greedy[ii]
    interval=temp[np.nonzero(temp)]

    #print(interval)
    kozachenko = continuous.get_h(interval, k=7)
    sdi_grd.append(kozachenko)
print('\n')

points=len(sdi)
sdi=np.array(sdi)/points
sdi_bys=np.array(sdi_bys)/points
sdi_grd=np.array(sdi_grd)/points

p=np.arange(0,points,1)


plt.plot(p,sdi)
plt.plot(p,sdi_grd)
plt.plot(p,sdi_bys)

idx=np.where(sdi!=-math.inf)
print(np.sum(sdi[idx]))
idx=np.where(sdi_grd!=-math.inf)
print(np.sum(sdi_grd[idx]))
idx=np.where(sdi_bys!=-math.inf)
print(np.sum(sdi_bys[idx]))

X = np.random.randn(10000, 2)
kozachenko = continuous.get_h(X, k=8)
print(kozachenko)
for i in range(ppo.len_episode):
    writer.add_scalars("IGI(t)",{'IGI-RL':idleness[i],'IGI-Greedy':idleness_greedy[i],'IGI-Bayes':idleness_bayes[i]},i)
plt.plot(all_ep_r)

if not os.path.exists('image'):
    os.makedirs('image')
















