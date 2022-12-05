import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

sys.path.append('..')
from Env.env import obsMap

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 下面老是报错 shape 不一致

from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical

import time

from patrol_policy.sy_patrol.sy_encoder import GraphEncoder
from patrol_policy.sy_patrol.sy_decoder import AttentionModel
from Env.env import Env
from Env.env import CONST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 999
np.random.seed(seed)
torch.manual_seed(seed)
run_name = f"multi_patrol{seed}_{int(time.time())}"
writer = SummaryWriter(log_dir=f"../runs/{run_name}")


class Policy(nn.Module):
    def __init__(self, env, args):
        hidden_dim = args.num_hidden
        action_dim = args.graph_size
        super(Policy, self).__init__()
        self.encoder = GraphEncoder(env, args).to(device)
        self.decoder = AttentionModel(env, args).to(device)
        self.fc1 = nn.Linear(5, hidden_dim).to(device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.ln = nn.LayerNorm([hidden_dim]).to(device)
        self.prob = nn.Linear(hidden_dim, action_dim).to(device)
        self.adj = env.adj
        self.select_type = "sampling"
        self.value = nn.Linear(action_dim * hidden_dim, 1).to(device)

    def forward(self, state, node_last_visit, decision_index, decision_flag, shift_action=None):

        state = torch.tensor(state).to(device)
        state = state.unsqueeze(1)
        state = torch.repeat_interleave(state, CONST.NUM_AGENTS, dim=1)  # (batch num_agent ,graph_size,3)

        batch_size = len(node_last_visit)
        graph_size = state.shape[-2]
        mask = np.zeros((batch_size, CONST.NUM_AGENTS, graph_size))
        zuobiao = torch.zeros((batch_size, CONST.NUM_AGENTS, 2)).to(device)
        idx = torch.zeros(batch_size, CONST.NUM_AGENTS).to(device)
        for i in range(batch_size):
            for j in range(CONST.NUM_AGENTS):
                zuobiao[i][j] = torch.tensor(obsMap.nodes[node_last_visit[i][j]]).to(device)
                idx[i][j] = torch.tensor(obsMap.patrol_nodes.index(node_last_visit[i][j])).to(device)
                z = idx[i][j]
                mask[i][j] = self.adj[int(z.cpu().detach().numpy())]
        mask = torch.tensor(mask, device=device, requires_grad=False)

        t_count = torch.zeros((batch_size, CONST.NUM_AGENTS, 1)).to(device)
        zuobiao = zuobiao.unsqueeze(2)

        zuobiao = torch.repeat_interleave(zuobiao, graph_size, dim=2)
        state = torch.cat((state, zuobiao), -1)

        node_embedding, state_value = self.encoder(state, node_last_visit)

        # (batch_size,num_agents,grpah_size,hidden_dim)
        n_embedding, s_value = self.encoder(state, node_last_visit)
        # node_embedding=self.fc1(state)  #encoder2
        X = node_embedding
        Y = X.contiguous().reshape(X.shape[0], X.shape[1], -1)
        # state_value = self.value(Y)

        selected_total, log_prob_total, shift_action_total = self.decoder(node_embedding,
                                                                          node_last_visit, decision_index,
                                                                          decision_flag, shift_action)

        return selected_total, log_prob_total, state_value, shift_action_total

    def _select_node(self, probs):
        select = []
        for i in range(len(probs)):
            probs_i = probs[i].squeeze()
            if self.select_type == 'greedy':
                _, selected = probs_i.max(1)

            elif self.select_type == 'sampling':

                selected = probs_i.multinomial(1)
                selected = selected.squeeze()

            select.append(selected)
        select = torch.stack(select).to(device)
        # select=select.reshape(probs.shape[0],probs.shape[1])
        return select


class PPO():
    def __init__(self, env, args):

        self.gamma = args.gamma
        self.policy_lr = args.lr
        self.train_episodes = args.train_episodes

        self.len_episode = CONST.LEN_EPISODE
        self.policy_update_steps = args.policy_update_steps
        self.batch_size = args.batch_size

        self.eps = 1e-8
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.env = env

        self.policy = Policy(self.env, args)

        self.policy.train()

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.policy_opt, mode='max',
                                                                    factor=args.factor, patience=args.patience,
                                                                    eps=1e-8)
        self.loss = nn.MSELoss()
        self.clip_coef = args.clip_coef
        self.state_buffer, self.action_buffer = [], []
        self.shift_action_buffer = []
        self.decision_index_buffer, self.decision_flag_buffer = [], []

        self.reward_buffer = []
        self.nodes_last_visit_buffer = []

        self.log_prob_buffer = []

        self.nodes_visit_buffer = []

    def store_transition(self, state, decision_index, decision_flag, action, reward, nodes_last_visit,
                         log_prob, nodes_visit):

        self.state_buffer.append(state)
        self.decision_index_buffer.append(decision_index)
        self.decision_flag_buffer.append(decision_flag)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

        self.nodes_last_visit_buffer.append(nodes_last_visit)

        self.log_prob_buffer.append(log_prob)

        self.nodes_visit_buffer.append(nodes_visit)

    def choose_node(self, state, nodes_last_list, decision_index, decision_flag):

        node_last_visit = []
        for i in range(CONST.NUM_AGENTS):
            node_last_visit.append(
                obsMap.get_key(nodes_last_list[i], obsMap.nodes))
        state = np.array(state)

        state = torch.tensor(state).to(device).unsqueeze(0)  # (1,graph_size,3)

        node_last_visit = np.array(node_last_visit)

        node_last_visit = node_last_visit[None, :]  # (1,num_agent)

        selected, log_prob, s_v, shift_action = self.policy(state, node_last_visit, decision_index, decision_flag)
        # (batch_size=1,num_agents) # (batch_size=1,num_agent,graph_size)
        selected = [item for item in selected[0]]

        # writer.add_scalar("current node", np.array(node_last_visit), episode * self.len_episode + t)
        log_prob_choose = []
        for i in range(CONST.NUM_AGENTS):
            log_prob_choose.append(log_prob[0][i][selected[i].cpu().detach().numpy()])
        nodes = []
        for i in range(CONST.NUM_AGENTS):
            cc = selected[i].cpu().detach().numpy()
            cc = np.array(cc, dtype=int)
            dd = obsMap.patrol_nodes[cc]
            nodes.append(obsMap.nodes[dd])
        return nodes, log_prob_choose, shift_action
        # return : choose的node : list: num_agent,2 , log_prob_choose : list : num_agent

    def adjust_Seq(self, nodes_rand, rand_Seq):  # all tensors
        rand_Seq = rand_Seq.detach().cpu().numpy()
        nodes_rand = np.array(nodes_rand)
        nodes = nodes_rand.copy()
        for i in range(CONST.NUM_AGENTS):
            nodes[i] = nodes_rand[np.where(rand_Seq == i)]
        return nodes

    # choose node: list: num_agent,2 log_prob_choose: list: num_agent
    # state:(1,graph_size,3)  nodes_last_visit:(1,num_agents)
    def greedy_choose_node(self, state, nodes_last_list):

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
            for j in nb_list:
                i = obsMap.patrol_nodes.index(j)
                if judge_time <= state[i][2] and not visitflag[j]:
                    judge_time = state[i][2]
                    selected = j
                    visitflag[j] = True

            nodes.append(obsMap.nodes[selected])
        log_prob_choose = [torch.tensor(0, device=device) for i in range(CONST.NUM_AGENTS)]
        return nodes, log_prob_choose

    def randchoose(self, state, nodes_last_list):
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
            i = np.random.randint(0, len(nb_list))
            selected = nb_list[i]

            nodes.append(obsMap.nodes[selected])
        log_prob_choose = [torch.tensor(0, device=device) for i in range(CONST.NUM_AGENTS)]
        return nodes, log_prob_choose

    def update_greedy(self):
        self.state_buffer.clear()
        self.decision_index_buffer.clear()
        self.decision_flag_buffer.clear()
        self.action_buffer.clear()
        self.shift_action_buffer.clear()
        self.reward_buffer.clear()
        self.log_prob_buffer.clear()

        self.nodes_last_visit_buffer.clear()
        self.nodes_visit_buffer.clear()

    def gae(self):
        state_next, node_next, shift_action_next = self.state_buffer[-1], self.nodes_last_visit_buffer[-1], \
                                                   self.shift_action_buffer[-1]
        decision_index_next, decision_flag_next = self.decision_index_buffer[-1], self.decision_flag_buffer[-1]
        st_buffer = []
        n_last_visit_buffer = []
        d_index_buffer = []
        d_flag_buffer = []
        shift_a_buffer = []
        for i in range(len(self.state_buffer)):
            if i < len(self.state_buffer) - 1:
                st_buffer.append(self.state_buffer[i])
                n_last_visit_buffer.append(self.nodes_last_visit_buffer[i])
                d_index_buffer.append(self.decision_index_buffer[i])
                d_flag_buffer.append(self.decision_flag_buffer[i])
                shift_a_buffer.append(self.shift_action_buffer[i])
        s = np.array(st_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.reward_buffer, np.float32)

        d_index_buffer = torch.stack(d_index_buffer)
        d_flag_buffer = torch.stack(d_flag_buffer)
        shift_a_ = torch.stack(shift_a_buffer)
        n = n_last_visit_buffer

        _, _, next_value, _ = self.policy(np.array(state_next)[None, :, :],
                                          [node_next], decision_index_next.unsqueeze(0),
                                          decision_flag_next.unsqueeze(0), shift_action_next)
        G__estimate_list = []
        next_value = next_value.mean().cpu().detach().numpy()
        for item in self.reward_buffer[::-1]:
            G_estimate_t = self.gamma * next_value + item
            next_value = G_estimate_t
            G__estimate_list.append(G_estimate_t)
        G__estimate_list.reverse()
        G_estimate_Tensor = torch.Tensor(G__estimate_list).to(device)
        G_estimate_Tensor = (G_estimate_Tensor - G_estimate_Tensor.mean()) / (G_estimate_Tensor.std() + 1e-8)
        _, _, value, _ = self.policy(s, n, d_index_buffer, d_flag_buffer, shift_a_.squeeze(1))
        advantage = G_estimate_Tensor.squeeze() - value.mean(axis=1).squeeze()
        return advantage, G_estimate_Tensor.squeeze()

    def update(self, episode_reward, batch_step):
        # del self.action_buffer[-1]
        advantage, returns = self.gae()
        st_buffer = []
        n_last_visit_buffer = []
        d_index_buffer = []
        d_flag_buffer = []
        shift_a_buffer = []
        for i in range(len(self.state_buffer)):
            if i < len(self.state_buffer) - 1:
                st_buffer.append(self.state_buffer[i])
                n_last_visit_buffer.append(self.nodes_last_visit_buffer[i])
                d_index_buffer.append(self.decision_index_buffer[i])
                d_flag_buffer.append(self.decision_flag_buffer[i])
                shift_a_buffer.append(self.shift_action_buffer[i])
        s = np.array(st_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        shift_a_ = torch.stack(shift_a_buffer)
        d_index_buffer = torch.stack(d_index_buffer)
        d_flag_buffer = torch.stack(d_flag_buffer)
        r = np.array(self.reward_buffer, np.float32)
        # log_prob_buffer=np.array(self.log_prob_buffer,np.float32)

        n = n_last_visit_buffer

        # adv=(advantage-advantage.mean())/(advantage.std()+1e-8)
        adv = advantage
        adv = torch.tensor(adv.detach()).to(device)
        # Reinforce
        G_list = []
        G_t = 0
        for item in self.reward_buffer[::-1]:
            G_t = self.gamma * G_t + item
            G_list.append(G_t)
        G_list.reverse()

        G_tensor = torch.tensor(G_list, dtype=torch.float).to(device)
        # baseline_tensor=torch.tensor(baseline_list,dtype=torch.float).to(device)

        _, log_prob, s_value, _ = self.policy(s, n, d_index_buffer, d_flag_buffer, shift_a_.squeeze(1))
        _, log_prob_new, value_new, _ = self.policy(s, n, d_index_buffer, d_flag_buffer, shift_a_.squeeze(1))
        # log_prob : (batch_size, num_agents, graph_size)
        # a: (batch_size,num_agents,2)
        # s_value: batch_size
        s_value = torch.tensor(s_value.detach()).to(device)
        s_value = s_value.mean(axis=1)
        log_prob = log_prob.cpu().detach().numpy()
        l = log_prob.shape[0]
        l_ = log_prob.shape[1]
        # prob/prob_old
        log_prob_buffer = []
        for i in range(l):
            log_prob_batch = []
            for k in range(l_):
                act = get_key(a[i][k], obsMap.nodes)
                j = obsMap.patrol_nodes.index(act)
                log_p = log_prob[i][k]
                log_prob_batch.append(log_p[j])
            log_prob_buffer.append(log_prob_batch)
        # log_prob_buffer: list(list) :(batch,num_agent)
        for _ in range(self.policy_update_steps):

            loss = 0
            _, log_prob_new, value_new, _ = self.policy(s, n, d_index_buffer, d_flag_buffer, shift_a_.squeeze(1))
            value_new = value_new.mean(axis=1)
            l = log_prob_new.shape[0]
            l_ = log_prob_new.shape[1]
            # prob/prob_old
            log_prob_new_buffer = []
            for i in range(l):
                log_prob_new_batch = []
                for k in range(l_):
                    act = get_key(a[i][k], obsMap.nodes)
                    j = obsMap.patrol_nodes.index(act)
                    log_p_new = log_prob_new[i][k]
                    log_prob_new_batch.append(log_p_new[j])
                log_prob_new_buffer.append(log_prob_new_batch)
            # log_prob_buffer: list(list) :(batch,num_agent)

            norm_factor = len(log_prob_new_buffer) * CONST.NUM_AGENTS
            policy_loss = 0
            for ad, log_p_new_batch, log_p_old_batch in zip(adv, log_prob_new_buffer, log_prob_buffer):
                for log_p_new, log_p_old in zip(log_p_new_batch, log_p_old_batch):
                    policy_loss += torch.max(-ad * (log_p_new - log_p_old).exp(),
                                             -ad * torch.clamp((log_p_new - log_p_old).exp(),
                                                               1 - self.clip_coef, 1 + self.clip_coef)) / norm_factor

            value_unclipped_loss = self.loss(value_new.squeeze(), returns)
            value_new_clipped = s_value + torch.clamp(value_new - s_value, -0.2, 0.2)
            value_clipped_loss = self.loss(value_new_clipped.squeeze(), returns)
            value_loss = torch.max(value_clipped_loss, value_unclipped_loss)

            # entropy_loss
            entropy = Categorical(logits=log_prob_new).entropy()
            entropy_loss = entropy.mean()
            loss = policy_loss - self.ent_coef * entropy_loss + self.vf_coef * value_clipped_loss

            # value loss

            self.policy_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_opt.step()

        self.scheduler.step(episode_reward)
        writer.add_scalar("loss/policy_loss", policy_loss, batch_step)
        writer.add_scalar("loss/value_loss", value_loss, batch_step)
        writer.add_scalar("loss/entropy_loss", entropy_loss, batch_step)
        writer.add_scalar("loss/loss", loss, batch_step)
        writer.add_scalar("learning rate", self.policy_opt.param_groups[0]["lr"], batch_step)

        del self.state_buffer[0:len(self.state_buffer) - 1]
        del self.decision_index_buffer[0:len(self.decision_index_buffer) - 1]
        del self.decision_flag_buffer[0:len(self.decision_flag_buffer) - 1]
        del self.shift_action_buffer[0:len(self.shift_action_buffer) - 1]
        self.action_buffer.clear()

        self.reward_buffer.clear()
        self.log_prob_buffer.clear()

        del self.nodes_last_visit_buffer[0:len(self.nodes_last_visit_buffer) - 1]

    def saveModel(self, filePath, per_save=False, episode=0, batchstep=0):

        if per_save == False:
            torch.save(self.policy.state_dict(), f"{filePath}/0agent.pt")

        else:
            torch.save(self.policy.state_dict(), f"{filePath}/0agent{episode}_{batchstep}.pt")

    def loadModel(self, filePath, cpu=1):

        if cpu == 1:
            self.policy.load_state_dict(torch.load(f"{filePath}/0agent100_501.pt", map_location=torch.device('cpu')))

        else:
            self.policy.load_state_dict(torch.load(f"{filePath}/0agent100_501.pt"))

        self.policy.eval()


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val[0] == value[0] and val[1] == value[1]:
            return key


class sy_trainer():
    def __init__(self, args):
        self.env = Env()
        self.ppo = PPO(self.env, args)
        # ppo.loadModel("../checkpoints")

        init_x = torch.zeros((1, 20, 3), device="cpu")

        init_x = torch.Tensor(init_x)

        self.all_ep_r = []

        # idleness=[]
        # idleness_greedy=[]
        self.idleness = []
        self.idleness_greedy = []
        self.batch_step = 0

    def train(self):
        for episode in range(self.ppo.train_episodes):

            state, nodes_last_visit, last_vertex, neighbor_nodes, flag = self.env.reset()

            nodes_last_goal = [nodes_last_visit[i] for i in range(CONST.NUM_AGENTS)]
            episode_reward = 0
            t0 = time.time()
            self.ppo.nodes_last_visit_buffer.clear()
            self.ppo.state_buffer.clear()
            self.ppo.decision_flag_buffer.clear()
            self.ppo.decision_index_buffer.clear()
            self.ppo.reward_buffer.clear()
            self.ppo.action_buffer.clear()
            self.ppo.shift_action_buffer.clear()

            change_reward = 0
            change_step = 0
            decision_index = -torch.ones((1, CONST.NUM_AGENTS)).to(device)

            for t in range(self.ppo.len_episode):

                if (episode + 1) % 1000 == 0:
                    self.env.render()
                decision_flag = torch.ones((1, CONST.NUM_AGENTS), dtype=bool).to(device)
                greedynodes = []

                for i in range(CONST.NUM_AGENTS):
                    decision_flag[0][i] = flag[i]
                if t == 0:
                    self.ppo.state_buffer.append(state)
                    self.ppo.decision_index_buffer.append(decision_index.squeeze())
                    self.ppo.decision_flag_buffer.append(decision_flag.squeeze())
                    rand_Seq = torch.randperm(CONST.NUM_AGENTS).to(device)
                    nodes_last_visit_rand = np.array(nodes_last_visit)
                    nodes_last_visit_rand = torch.tensor(nodes_last_visit_rand, device=device, dtype=torch.float32)
                    nodes_last_visit_rand = nodes_last_visit_rand[rand_Seq, :]
                    self.ppo.nodes_last_visit_buffer.append([get_key(nodes_last_visit_rand[i].squeeze(),
                                                                     obsMap.nodes) for i in range(CONST.NUM_AGENTS)])

                    nodes_rand, log_prob, shift_action = self.ppo.choose_node(state, nodes_last_visit_rand,
                                                                              decision_index,
                                                                              decision_flag)
                    nodes = self.ppo.adjust_Seq(nodes_rand, rand_Seq)
                    # shift_action = torch.zeros((1, CONST.NUM_AGENTS, CONST.NUM_AGENTS, 2)).to(device)
                    self.ppo.shift_action_buffer.append(shift_action)
                    # nodes, log_prob = ppo.greedy_choose_node(state, nodes_last_visit)
                    # nodes, log_prob = ppo.randchoose(state, nodes_last_visit)
                elif any(decision_flag[0]):
                    rand_Seq = torch.randperm(CONST.NUM_AGENTS).to(device)
                    nodes_last_visit_rand = np.array(nodes_last_visit)
                    nodes_last_visit_rand = torch.tensor(nodes_last_visit_rand, device=device, dtype=torch.float32)
                    nodes_last_visit_rand = nodes_last_visit_rand[rand_Seq, :]
                    self.ppo.store_transition(state, decision_index.squeeze(), decision_flag.squeeze(), nodes_rand,
                                              change_reward,
                                              [get_key(nodes_last_visit_rand[i].squeeze(),
                                                       obsMap.nodes) for i in range(CONST.NUM_AGENTS)], log_prob,
                                              [get_key(torch.Tensor(n_last_visit[i]).squeeze(), obsMap.nodes)
                                               for i in range(CONST.NUM_AGENTS)])

                    nodes_rand, log_prob, shift_action = self.ppo.choose_node(state, nodes_last_visit_rand,
                                                                              decision_index,
                                                                              decision_flag)
                    nodes = self.ppo.adjust_Seq(nodes_rand, rand_Seq)
                    # shift_action=torch.zeros((1,CONST.NUM_AGENTS,CONST.NUM_AGENTS,2)).to(device)
                    self.ppo.shift_action_buffer.append(shift_action)
                    # nodes,log_prob=ppo.greedy_choose_node(state,nodes_last_visit)
                    # nodes, log_prob = ppo.randchoose(state, nodes_last_visit)
                    change_reward = 0


                else:
                    nodes = []
                    for i in range(CONST.NUM_AGENTS):
                        nodes.append(obsMap.nodes[int(decision_index[0][i])])

                a = time.time()

                decision_index = [get_key(nodes[i], obsMap.nodes) for i in range(CONST.NUM_AGENTS)]
                decision_index = np.array(decision_index)
                decision_index = torch.tensor(decision_index, device=device).unsqueeze(0)  # (1,num_agents)
                next_state, agent_pos_list, current_map_state, nb_nodes, local_heatmap_list, mini_map, \
                shared_reward, flag, last_vertex_list, n_last_visit, visit_flag, done = self.env.step(
                    nodes, nodes_last_visit, last_vertex, state)
                if episode == 1:
                    self.idleness.append(self.env.avg_idleness)
                elif episode == 0:
                    self.idleness_greedy.append(self.env.avg_idleness)
                last_vertex = [last_vertex_list[i] for i in range(CONST.NUM_AGENTS)]
                nodes_last_goal = [nodes[i] for i in range(CONST.NUM_AGENTS)]
                change_reward += shared_reward  # change_reward
                if any(decision_flag[0]):
                    change_step += 1

                state = next_state
                nodes_last_visit = n_last_visit

                episode_reward += shared_reward

                if change_step % self.ppo.batch_size == 0 and any(decision_flag[0]) and len(
                        self.ppo.reward_buffer) or t == self.ppo.len_episode:
                    cc = 1
                    self.ppo.saveModel("../checkpoints", True, episode, self.batch_step)
                    # print("Store Successfully !")
                    # if batch_step<3:
                    self.ppo.update(episode_reward, self.batch_step)
                    # else:
                    #    ppo.update_greedy()
                    self.batch_step += 1

            if episode == 0:
                self.all_ep_r.append(episode_reward)
            else:
                self.all_ep_r.append(self.all_ep_r[-1] * 0.9 + episode_reward * 0.1)
            writer.add_scalar("episode_reward", episode_reward, episode)

            if (episode + 1) % 10 == 0:
                print(self.all_ep_r)
            print(
                'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f} | Idleness : {:.4f} | Global Idleness : {:.4f}'.format(
                    episode, self.ppo.train_episodes, episode_reward,
                    time.time() - t0, self.env.avg_idleness, self.env.avg_i_sum / CONST.LEN_EPISODE
                )
            )

            if episode % 50 == 0:
                self.ppo.saveModel("../checkpoints", True)
            # plt.show()

        tt = np.arange(0, self.ppo.len_episode, 1)
        idleness = np.array(self.idleness) / 25
        idleness_greedy = np.array(self.idleness_greedy) / 25
        plt.plot(tt, idleness)
        plt.plot(tt, idleness_greedy)
        plt.show()

        plt.plot(self.all_ep_r)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join(['attention_with_ppo', 'single_patrol'])))
