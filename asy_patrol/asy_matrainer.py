

import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch import nn

from env import obsMap
from d2l import torch as d2l
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  #

from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical


import time


from asy_encoder import GraphEncoder
from asy_decoder import AttentionModel
from env import Env



from env import CONST



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 1
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
        self.value = nn.Linear(25 * hidden_dim, 1).to(device)



    def get_shift_action(self,decision_flag,nodes,nodes_last_visit):

        shift_act=torch.zeros((1,CONST.NUM_AGENTS,CONST.NUM_AGENTS,2)).to(device)
        shift_act[:,0,0,0]=1


        batch_size=decision_flag.shape[0]
        agent_id=torch.arange(0,CONST.NUM_AGENTS).unsqueeze(0)
        agent_id=torch.repeat_interleave(agent_id,batch_size,dim=0)

        if nodes==None:
            return shift_act,nodes_last_visit,agent_id,decision_flag
        nodes=np.array(nodes)
        nodes=nodes.reshape(batch_size,CONST.NUM_AGENTS,2)

        for b in range(batch_size):
            for agent in reversed(range(0,CONST.NUM_AGENTS-1)):
                if decision_flag[b][agent]:
                    for agt_ in range(agent+1,CONST.NUM_AGENTS):
                        if decision_flag[b][agt_-1]==True and decision_flag[b][agt_]==False:

                            tmp=decision_flag[b][agt_-1].clone()
                            decision_flag[b][agt_-1]=decision_flag[b][agt_].clone()
                            decision_flag[b][agt_]=tmp

                            tmp=nodes_last_visit[b][agt_-1].copy()
                            nodes_last_visit[b][agt_ - 1]=nodes_last_visit[b][agt_].copy()
                            nodes_last_visit[b][agt_]=tmp

                            tmp= nodes[b][agt_-1].copy()
                            nodes[b][agt_-1]=nodes[b][agt_].copy()
                            nodes[b][agt_]=tmp

                            tmp=agent_id[b][agt_-1].clone()
                            agent_id[b][agt_ - 1]=agent_id[b][agt_].clone()
                            agent_id[b][agt_]=tmp


        for b in range(batch_size):
            for agent in range(CONST.NUM_AGENTS-1):
                if not decision_flag[b][agent]:
                    shift_act[:,:,agent+1,:]=torch.tensor(nodes[b][agent],dtype=torch.float32,device=device)



        return shift_act,nodes_last_visit,agent_id,decision_flag


    def forward(self, state, node_last_visit,decision_index,decision_flag,nodes=None,shift_action=None,agent_id=None):

        batch_size = len(node_last_visit)
        graph_size = state.shape[-2]

        if agent_id==None:
           agent_id = torch.arange(0, CONST.NUM_AGENTS).unsqueeze(0)
           agent_id = torch.repeat_interleave(agent_id, batch_size, dim=0)

        if shift_action ==None:
            shift_action,node_last_visit,agent_id,decision_flag=self.get_shift_action(decision_flag,nodes,node_last_visit)

        state=torch.tensor(state).to(device)
        if state.ndimension()<4:
            state=state.unsqueeze(1)
        if state.shape[1]<CONST.NUM_AGENTS:
          state=torch.repeat_interleave(state, CONST.NUM_AGENTS, dim=1)  # (batch num_agent ,graph_size,3)


        mask = np.zeros((batch_size, CONST.NUM_AGENTS, graph_size))
        #zuobiao=torch.zeros((batch_size,CONST.NUM_AGENTS,2)).to(device)
        idx = torch.zeros(batch_size, CONST.NUM_AGENTS).to(device)
        node_last_visit=np.array(node_last_visit)
        node_last_visit=node_last_visit.reshape(batch_size,CONST.NUM_AGENTS)
        zuobiao = torch.zeros((batch_size, CONST.NUM_AGENTS, 2)).to(device)
        for i in range(batch_size):
            for j in range(CONST.NUM_AGENTS):
                zuobiao[i][j]=torch.tensor(obsMap.nodes[node_last_visit[i][j]]).to(device)
                a=node_last_visit[i][j]
                idx[i][j] = torch.tensor(obsMap.patrol_nodes.index(a)).to(device)
                z = idx[i][j]
                mask[i][j] = self.adj[int(z.cpu().detach().numpy())]
        mask = torch.tensor(mask, device=device, requires_grad=False)

        t_count=torch.zeros((batch_size,CONST.NUM_AGENTS,1)).to(device)
        zuobiao=zuobiao.unsqueeze(2)

        zuobiao=torch.repeat_interleave(zuobiao,graph_size,dim=2)
        if state.shape[-1]<5:
         state=torch.cat((state,zuobiao),-1)

        node_embedding, state_value = self.encoder(state, node_last_visit)
        #node_embedding, state_value = self.encoder(zuobiao, node_last_visit) # (batch_size,num_agents,grpah_size,hidden_dim)

        #node_embedding=self.fc1(state)  #encoder2
        X=node_embedding
        Y = X.contiguous().reshape(X.shape[0], X.shape[1], -1)
        #state_value = self.value(Y)

        selected_total, log_prob_total,shift_action_total = self.decoder(node_embedding,
                                                      node_last_visit,decision_index,decision_flag,agent_id,shift_action)

        policy_input={"state":state,"node last visit":node_last_visit,"decision_flag":
            decision_flag,"agent_id":agent_id,"shift_action_total":shift_action_total}
        return selected_total, log_prob_total, state_value,policy_input

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


class PPO():
    def __init__(self, env):

        self.gamma = 0.98
        self.policy_lr = 0.000025
        self.train_episodes = 800
        self.test_episodes = 100
        self.len_episode = CONST.LEN_EPISODE
        self.policy_update_steps = 5
        self.batch_size =40
        self.mini_batch=5
        # 每10次更新网络
        self.METHOD = [
            dict(name='kl_pen', kl_target=0.01, lam=0.5),  # KL penalty
            dict(name='clip', epsilon=0.2),  # Clipped surrogate objective, find this is better
        ][1]
        self.eps = 1e-8
        self.ent_coef = 0.01
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
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.policy_opt,mode='max',factor=0.3,patience=50,eps=1e-8)
        self.loss = nn.MSELoss()

        self.state_buffer, self.action_buffer = [], []
        self.shift_action_buffer=[]
        self.decision_index_buffer, self.decision_flag_buffer = [], []


        self.reward_buffer = []
        self.nodes_last_visit_buffer = []

        self.log_prob_buffer = []
        self.G_buffer = []
        self.agent_id_buffer=[]
        self.nodes_visit_buffer = []



    def store_actionAndreward(self,action,reward):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def store_policy_input(self,policy_input):
        state,nodes_last_visit,decision_flag,agent_id,shift_action_total=policy_input["state"],\
                policy_input["node last visit"],policy_input["decision_flag"],\
                policy_input["agent_id"],policy_input["shift_action_total"]
        self.state_buffer.append(state)
        nodes_last_visit=torch.tensor(nodes_last_visit).to(device)
        self.nodes_last_visit_buffer.append(nodes_last_visit)
        self.decision_flag_buffer.append(decision_flag.squeeze())
        self.agent_id_buffer.append(agent_id.squeeze())
        self.shift_action_buffer.append(shift_action_total)

    # state (batch_size=1,num_agent,graph_size,3)
    # nodes_last_list : ( batch_size=1,num_agent) np.array
    # decision_index  : (batch_size=1, num_agent)
    # decision_flag  : (batch_size=1,  num_agent) bool
    def choose_node(self, state, nodes_last_list,decision_index,decision_flag,nodes):

        node_last_visit = []
        for i in range(CONST.NUM_AGENTS):
            node_last_visit.append(
                obsMap.get_key(nodes_last_list[i], obsMap.nodes))
        state=np.array(state)

        state=torch.tensor(state).to(device).unsqueeze(0) # (1,graph_size,3)

        node_last_visit=np.array(node_last_visit)

        node_last_visit=node_last_visit[None,:]
        selected, log_prob, s_v,policy_input = self.policy(state, node_last_visit,decision_index,decision_flag,nodes)

        self.store_policy_input(policy_input)

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
        return nodes, log_prob_choose




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

            selected = 0
            judge_time = 0
            for j in nb_list:
               i = obsMap.patrol_nodes.index(j)
               if judge_time <= state[i][2] :
                  judge_time = state[i][2]
                  selected = j
                  #visitflag[j]=True

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
        self.agent_id_buffer.clear()
        self.nodes_last_visit_buffer.clear()
        self.nodes_visit_buffer.clear()

    def gae(self):
        state_next,node_next,shift_action_next,agent_id_next=self.state_buffer[-1],self.nodes_last_visit_buffer[-1],\
                                               self.shift_action_buffer[-1],self.agent_id_buffer[-1]
        decision_flag_next=self.decision_flag_buffer[-1]
        st_buffer=[]
        n_last_visit_buffer=[]
        d_index_buffer=[]
        d_flag_buffer=[]
        shift_a_buffer=[]
        agt_id_buffer=[]
        for i in range(len(self.state_buffer)):
            if i <len(self.state_buffer)-1:
                st_buffer.append(self.state_buffer[i])
                n_last_visit_buffer.append(self.nodes_last_visit_buffer[i].cpu().numpy())

                d_flag_buffer.append(self.decision_flag_buffer[i])
                shift_a_buffer.append(self.shift_action_buffer[i])
                agt_id_buffer.append(self.agent_id_buffer[i])

        s = torch.stack(st_buffer)
        s=s.detach().cpu().numpy().squeeze(1)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.reward_buffer, np.float32)
        agt_id_buffer=torch.stack(agt_id_buffer)
        d_flag_buffer=torch.stack(d_flag_buffer)
        shift_a_=torch.stack(shift_a_buffer)
        n = n_last_visit_buffer

        node_next=node_next.detach().cpu().numpy()
        if agent_id_next.ndimension()<1:
            agent_id_next=agent_id_next.unsqueeze(0)
        _,_,next_value,_=self.policy(state_next,
                                   [node_next],None,
                                   decision_flag_next.unsqueeze(0),None,shift_action_next,agent_id_next)
        G__estimate_list=[]
        next_value=next_value.mean().cpu().detach().numpy()
        for item in self.reward_buffer[::-1]:
            G_estimate_t = self.gamma * next_value + item
            next_value=G_estimate_t
            G__estimate_list.append(G_estimate_t)
        G__estimate_list.reverse()
        G_estimate_Tensor=torch.Tensor(G__estimate_list).to(device)
        G_estimate_Tensor=(G_estimate_Tensor-G_estimate_Tensor.mean())/(G_estimate_Tensor.std()+1e-8)
        _,_,value,_=self.policy(s,n,d_index_buffer,d_flag_buffer,None,shift_a_.squeeze(1),agt_id_buffer.squeeze())
        advantage=G_estimate_Tensor.squeeze()-value.mean(axis=1).squeeze()
        return advantage,G_estimate_Tensor.squeeze()


    def update(self):
        #del self.action_buffer[-1]
        advantage,returns=self.gae()
        st_buffer = []
        n_last_visit_buffer = []
        agt_id_buffer=[]
        d_index_buffer=[]
        d_flag_buffer=[]
        shift_a_buffer=[]
        for i in range(len(self.state_buffer)):
            if i < len(self.state_buffer) - 1:
                st_buffer.append(self.state_buffer[i])
                n_last_visit_buffer.append(self.nodes_last_visit_buffer[i].cpu().numpy())

                d_flag_buffer.append(self.decision_flag_buffer[i])
                agt_id_buffer.append(self.agent_id_buffer[i])
                shift_a_buffer.append(self.shift_action_buffer[i])
        s = torch.stack(st_buffer)
        s = s.detach().cpu().numpy().squeeze(1)
        a = np.array(self.action_buffer, np.float32)
        shift_a_=torch.stack(shift_a_buffer)
        agt_id_buffer = torch.stack(agt_id_buffer)
        d_flag_buffer=torch.stack(d_flag_buffer)
        r = np.array(self.reward_buffer, np.float32)
        # log_prob_buffer=np.array(self.log_prob_buffer,np.float32)

        n = n_last_visit_buffer




        #adv=(advantage-advantage.mean())/(advantage.std()+1e-8)
        adv=advantage
        adv = torch.tensor(adv.detach()).to(device)
        # Reinforce
        G_list = []
        G_t = 0
        for item in self.reward_buffer[::-1]:
            G_t = self.gamma * G_t + item
            G_list.append(G_t)
        G_list.reverse()
        #G_list=self.G_buffer

        G_tensor = torch.tensor(G_list, dtype=torch.float).to(device)
        #baseline_tensor=torch.tensor(baseline_list,dtype=torch.float).to(device)
        '''loss = 0
        for g, log_prob in zip(G_tensor,self.log_prob_buffer):
            loss += -g * log_prob'''
        _, log_prob, s_value,_ = self.policy(s, n,None,d_flag_buffer,None,
                                             shift_a_.squeeze(1),agt_id_buffer.squeeze())

        # log_prob : (batch_size, num_agents, graph_size)
        # a: (batch_size,num_agents,2)
        # s_value: batch_size
        s_value=torch.tensor(s_value.detach()).to(device)
        s_value=s_value.mean(axis=1)
        log_prob = log_prob.cpu().detach().numpy()
        l = log_prob.shape[0]
        l_=log_prob.shape[1]
        # prob/prob_old
        log_prob_buffer = []
        for i in range(l):
            log_prob_batch=[]
            for k in range(l_):
                act = get_key(a[i][k], obsMap.nodes)
                j = obsMap.patrol_nodes.index(act)
                log_p = log_prob[i][k]
                log_prob_batch.append(log_p[j])
            log_prob_buffer.append(log_prob_batch)
        # log_prob_buffer: list(list) :(batch,num_agent)
        for _ in range(self.policy_update_steps):
                '''
                for start in range(0,self.mini_batch,self.batch_size):
                end=start+self.mini_batch
                s_mini=s[start:end]
                n_mini=n[start:end]
                a_mini=a[start:end]
                G_tensor_mini=G_tensor[start:end]
                '''
                loss = 0
                _, log_prob_new, value_new,_ = self.policy(s, n,d_index_buffer,d_flag_buffer,None,
                                                           shift_a_.squeeze(1),agt_id_buffer.squeeze())
                value_new=value_new.mean(axis=1)
                l = log_prob_new.shape[0]
                l_=log_prob_new.shape[1]

                log_prob_new_buffer = []
                for i in range(l):
                    log_prob_new_batch=[]
                    for k in range(l_):
                       act = get_key(a[i][k], obsMap.nodes)
                       j = obsMap.patrol_nodes.index(act)
                       log_p_new = log_prob_new[i][k]
                       log_prob_new_batch.append(log_p_new[j])
                    log_prob_new_buffer.append(log_prob_new_batch)
                # log_prob_buffer: list(list) :(batch,num_agent)


                #loss2: ppo_loss
                # policy_loss
                norm_factor=len(log_prob_new_buffer)*CONST.NUM_AGENTS
                policy_loss=0
                for ad, log_p_new_batch, log_p_old_batch in zip(adv, log_prob_new_buffer, log_prob_buffer):
                    for log_p_new,log_p_old in zip(log_p_new_batch,log_p_old_batch):
                        policy_loss += torch.max(-ad*(log_p_new - log_p_old).exp(),
                                         -ad * torch.clamp((log_p_new - log_p_old).exp(), 0.95, 1.05))/norm_factor
                #value_loss

                value_unclipped_loss=self.loss(value_new.squeeze(),returns)
                value_new_clipped=s_value+torch.clamp(value_new-s_value,-0.2,0.2)
                value_clipped_loss=self.loss(value_new_clipped.squeeze(),returns)
                value_loss=torch.max(value_clipped_loss,value_unclipped_loss)

                #entropy_loss
                entropy = Categorical(logits=log_prob_new).entropy()
                entropy_loss = entropy.mean()
                loss= policy_loss-self.ent_coef*entropy_loss  +self.vf_coef*value_clipped_loss


                # value loss

                self.policy_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_opt.step()


        self.scheduler.step(episode_reward)
        writer.add_scalar("loss/policy_loss", policy_loss, batch_step)
        writer.add_scalar("loss/value_loss",value_loss,batch_step)
        writer.add_scalar("loss/entropy_loss",entropy_loss,batch_step)
        writer.add_scalar("loss/loss",loss,batch_step)
        writer.add_scalar("learning rate", self.policy_opt.param_groups[0]["lr"],batch_step)
        '''for i in range(len(r)):  #td是时序上的，每个时间点，到一个点，对应的点是nodes_last_list[i]
            if i<len(r)-1:
                goal_list[obsMap.patrol_nodes.index(n[i+1])]+=r[i]
            else:
                goal_list[obsMap.patrol_nodes.index(kk)] += r[i]
            goal_list2[obsMap.patrol_nodes.index(n[i])] += td[i]'''



        del self.state_buffer[0:len(self.state_buffer)-1]
        del self.decision_flag_buffer[0:len(self.decision_flag_buffer)-1]
        del self.shift_action_buffer[0:len(self.shift_action_buffer)-1]
        del self.agent_id_buffer[0:len(self.agent_id_buffer)-1]
        self.action_buffer.clear()

        self.reward_buffer.clear()
        self.log_prob_buffer.clear()
        self.G_buffer.clear()

        del self.nodes_last_visit_buffer[0:len(self.nodes_last_visit_buffer)-1]

    def buffer_clear(self):
        self.nodes_last_visit_buffer.clear()
        self.state_buffer.clear()
        self.decision_flag_buffer.clear()
        self.decision_index_buffer.clear()
        self.reward_buffer.clear()
        self.action_buffer.clear()
        self.shift_action_buffer.clear()
        self.agent_id_buffer.clear()

    def saveModel(self, filePath, per_save=False, episode=0,batchstep=0):

        if per_save == False:
            torch.save(self.policy.state_dict(), f"{filePath}/asynchronous_policy_4agent.pt")

        else:
            torch.save(self.policy.state_dict(), f"{filePath}/asynchronous_policy_4agent{episode}_{batchstep}.pt")

    def loadModel(self, filePath, cpu=1):

        if cpu == 1:
            self.policy.load_state_dict(torch.load(f"{filePath}/asynchronous_4agent_b.pt", map_location=torch.device('cpu')))

        else:
            self.policy.load_state_dict(torch.load(f"{filePath}/asynchronous_4agent_b.pt"))

        self.policy.eval()


# main


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val[0] == value[0] and val[1] == value[1]:
            return key



env = Env()
ppo = PPO(env)
ppo.loadModel("../checkpoints")
run_name = f"multi_patrol{seed}_{int(time.time())}"
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

idleness=[]
idleness_greedy=[]
idleness_rand=[]
batch_step = 0
for episode in range(ppo.train_episodes):
    # state=env.reset()
    state, nodes_last_visit, last_vertex, neighbor_nodes, flag = env.reset()
    nodes_last_goal = [nodes_last_visit[i] for i in range(CONST.NUM_AGENTS)]
    episode_reward = 0
    t0 = time.time()


    ppo.buffer_clear()
    change_reward = 0
    change_step = 0
    decision_index = -torch.ones((1, CONST.NUM_AGENTS)).to(device)


    for t in range(ppo.len_episode):

        if (episode + 1) % 1000 == 0:
            env.render()


        decision_flag=torch.ones((1,CONST.NUM_AGENTS),dtype=bool).to(device)
        greedynodes = []


        for i in range(CONST.NUM_AGENTS):

            decision_flag[0][i]=flag[i]
        if t==0:

            nodes, log_prob= ppo.choose_node(state, nodes_last_visit, decision_index, decision_flag,None)


        elif any(decision_flag[0]):
            ppo.store_actionAndreward(nodes,change_reward)

            nodes,log_prob=ppo.choose_node(state,nodes_last_visit,decision_index,decision_flag,nodes)

            change_reward=0
        # choose的node: list: num_agent,2 log_prob_choose: list: num_agent
        # state:(1,graph_size,3)  nodes_last_visit:(1,num_agents)

        else:
            nodes = []
            for i in range(CONST.NUM_AGENTS):
                nodes.append(obsMap.nodes[int(decision_index[0][i])])




        a = time.time()

        # greedy_env_step:


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
        if change_step % ppo.batch_size == 0 and any(decision_flag[0]) and len(ppo.reward_buffer) or t == ppo.len_episode :
            cc = 1
            ppo.saveModel("../checkpoints", True, episode, batch_step)
            #print("Store Successfully !")
            #if batch_step<3:
            ppo.update()
            #else:
            #    ppo.update_greedy()
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



    # plt.show()

tt=np.arange(0,ppo.len_episode,1)
idleness=np.array(idleness)/20
idleness_greedy=np.array(idleness_greedy)/20
idleness_rand=np.array(idleness_rand)/20
plt.plot(tt,idleness)
plt.plot(tt,idleness_greedy)
plt.plot(tt,idleness_rand)
plt.show()


plt.plot(all_ep_r)
if not os.path.exists('image'):
    os.makedirs('image')
plt.savefig(os.path.join('image', '_'.join(['attention_with_ppo', 'single_patrol'])))















