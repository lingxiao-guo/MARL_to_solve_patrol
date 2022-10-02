import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math
from env import obsMap


from typing import NamedTuple


from env import CONST
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class AttentionModelFixed():

    def __init__(self,node_embeddings,context_node_projected,glimpse_key,glimpse_val,logit_key):
        self.node_embeddings=node_embeddings
        self.context_node_projected=context_node_projected
        self.glimpse_key=glimpse_key
        self.glimpse_val=glimpse_val
        self.logit_key=logit_key


class AttentionModel(nn.Module):

    def __init__(self,env,encoder_dim=32,hidden_dim=32,tanh_clipping=10,action_dim=2,
                 normalization='batch',n_heads=4,norm_shape=32,select_type='sampling',checkpoint_encoder=False):
        super(AttentionModel,self).__init__()
        self.temp=1
        self.encoder_dim=encoder_dim
        self.hidden_dim=hidden_dim
        self.tanh_clipping=tanh_clipping
        self.adj=env.adj
        self.n_heads=n_heads
        self.checkpoint_encoder=checkpoint_encoder
        self.select_type=select_type
        self.ln=nn.LayerNorm(norm_shape)
        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(encoder_dim, 3 * encoder_dim, bias=False).to(device)
        self.project_fixed_context = nn.Linear(encoder_dim, encoder_dim, bias=False).to(device)
        self.project_step_context=nn.Linear(encoder_dim,encoder_dim,bias=False).to(device)
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(encoder_dim, encoder_dim, bias=False).to(device)
        self.action_embedding=nn.Linear(action_dim,hidden_dim).to(device)
        self.self_attn=SelfAttention_()
        self.attn=Attention_()

    def forward(self,encoder_output,node_last_list,decision_index,decision_flag,shift_action=None):

        decoder_input=encoder_output


        log_prob,selected,shift_action_total=self._get_log_p(encoder_output,self.adj,node_last_list,shift_action)


        '''if len(selected)==1:
            selected=nb_nodes[selected]
        else:
            sl = []
            for i in range(len(selected)):
                sl.append(nb_nodes[i][selected[i]])
            # only integer tensors of a single element can be converted to an index
            selected=sl
        '''

        return selected,log_prob,shift_action_total

    def _precompute(self,encoder_output,action_embed):
        #encoder_output.shape():batch_size,num_agents,graph_size,hidden_dim
        action_embed=action_embed.unsqueeze(2)
        graph_size=encoder_output.shape[2]
        action_embed=torch.repeat_interleave(action_embed,graph_size,dim=2)
        encoder_output = self.ln(encoder_output + self.attn(encoder_output, action_embed, action_embed))
        graph_embed=torch.mean(encoder_output,axis=-2)
        graph_embed=graph_embed.to(device)
        fixed_context=self.project_fixed_context(graph_embed)[:,:,None,:]
        #encoder_output=torch.Tensor(encoder_output)
        # (batch_size,num_agents,1,hidden_dim)

        glimpse_key_fixed,glimpse_val_fixed,logit_key_fixed=\
            self.project_node_embeddings(encoder_output).chunk(3,dim=-1)
        # batch_size,num_agents,graph_size,hidden_dim
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed),
            self._make_heads(glimpse_val_fixed),
            logit_key_fixed.contiguous()
        )

        return AttentionModelFixed(encoder_output, fixed_context, *fixed_attention_node_data)

    def _make_heads(self, v):

          return (
              v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1).permute(3, 0, 1, 2, 4)
          )  # (n_heads,batch_size,num_agents,graph_size,head_hidden_dim)

    def _get_log_p(self,decoder_input,adj,node_last_list,shift_action=None,normalize=True):

          #Compute query=context node embedding

          node_last_list=np.array(node_last_list)

          action_total=[]
          batch_size=decoder_input.shape[0]
          graph_size=decoder_input.shape[-2]
          log_prob_total=torch.zeros((batch_size,CONST.NUM_AGENTS,graph_size)).to(device)
          istrain=True
          if shift_action==None:
             istrain=False
             shift_action=torch.zeros((batch_size,CONST.NUM_AGENTS,CONST.NUM_AGENTS,2)).to(device)
             shift_action[:,0,0,0]=1
          visitflag=[False for i in range(graph_size)]

          for agent in range(CONST.NUM_AGENTS):

              action_embed = self.action_embedding(shift_action[:, agent, :, :])
              # (batch_size,num_agent,hidden_dim)
              action_embed = self.self_attn(action_embed, action_embed, action_embed)
              fixed = self._precompute(decoder_input,action_embed)
              i, recent_node = self.get_recent_node(fixed.node_embeddings[:,agent,:,:], node_last_list[:,agent])

              query = fixed.context_node_projected[:,agent,:,:] + \
                      self.project_step_context(recent_node)
              # query:(batch_size,1,hidden_dim)


              #query=self.attn(query,action_embed,action_embed)
              #query=self.ln(query+self.attn(query,action_embed,action_embed))

              # shift_action: (batch_size,num_agent,action_dim)
              glimpse_K, glimpse_V, logit_K = fixed.glimpse_key[:,:,agent,:,:], fixed.glimpse_val[:,:,agent,:,:], \
                                              fixed.logit_key[:,agent,:,:]
              glimpse_Q = query .view(query.size(0), self.n_heads, 1, query.size(2) // self.n_heads).permute(1, 0, 2, 3)

              # if batch_size==1:
              # glimpse_Q:(n_heads,batch_size,1,feature_dim//n_heads)
              # glimpse_V:(n_heads,batch_size,graph_size,feature_dim//n_heads))
              # glimpse_K:(n_heads,batch_size,graph_size,feature_dim//n_heads))
              # logit_K:(batch_size,graph_size,hidden_dim)
              dim = glimpse_K.size(-1)
              weight = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(dim)
              # (n_heads,batch_size,1,graph_size)
              # i:list
              adj_m = []
              for k in range(len(i)):
                  c=i[k]
                  d=self.adj[c]
                  adj_m.append(self.adj[i[k]])
              adj_m = np.array(adj_m, dtype=int)[:, None, :]
              adj = adj_m
              np.expand_dims(adj_m, 0).repeat(glimpse_K.shape[0], axis=0)
              # adj_m:(n_heads,batch_size,1,graph_size)
              # adj:(batch_size,1,graph_size)
              adj_m = torch.tensor(adj_m).to(device)
              adj = torch.tensor(adj).to(device)
              zero_vec = -9e15 * torch.ones_like(weight)
              weight = torch.where(adj_m > 0, weight, zero_vec)
              score = torch.matmul(nn.functional.softmax(weight, dim=-1), glimpse_V)
              # (n_heads,batch_size,1,feature_dim//n_heads)
              final_Q = self.project_out(
                  score.permute(1, 2, 0, 3).contiguous().view(score.shape[1], score.shape[2],
                                                              self.n_heads * score.shape[-1]))
              # (batch_size,1,hidden_dim)
              logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)) / math.sqrt(dim)
              logits = torch.tanh(logits) * self.tanh_clipping
              zero_vec = -9e15 * torch.ones_like(logits)
              logits = torch.where(adj > 0, logits, zero_vec)
              log_prob = torch.log_softmax(logits / self.temp, dim=-1)
              log_prob_total[:,agent,:]=log_prob.squeeze()
              log_prob=log_prob.clone()
              prob = log_prob.exp()
              selected,visitflag = self._select_node(prob,visitflag)
              action_total.append(selected)


              if agent<CONST.NUM_AGENTS-1 and not istrain:
                  shift_action[:,agent+1,:,:]=shift_action[:,agent,:,:].clone()
                  for i in range(selected.shape[0]):
                     c=1

                     shift_action[i,agent+1,agent+1,:]=torch.tensor(obsMap.nodes
                                                                [obsMap.patrol_nodes[selected[i].clone()]] ).to(device)
                     #默认selected: (batch_size)

                     #shift_action=shift_action.contiguous()

          action_total=torch.stack(action_total).to(device)
          action_total=action_total.reshape(batch_size,CONST.NUM_AGENTS)
          return log_prob_total,action_total,shift_action





    def get_recent_node(self,node_embedding,node_last_list):
        i=[]
        recent_node=[]
        for k in range(len(node_last_list)):  #k:0~49
          i.append(obsMap.patrol_nodes.index(node_last_list[k]))
          recent_node.append(node_embedding[k,i[k],:])
        recent_node=torch.stack(recent_node).unsqueeze(1)
        return i,recent_node

    def _select_node(self, probs, visitflag):
          #visitflag=[False for i in range(len(visitflag))]
          vf_copy = visitflag.copy()
          probs = probs.squeeze()
          probs_copy = probs.clone()
          for i in range(len(visitflag)):
              if visitflag[i]:
                  probs[i] = 0
          flag = False
          # if test, delete following "#", which means two agents wouldn't choose the same points.
          # It benefits to the performance

          for i in range(len(probs)):
              if probs[i] != 0:
                  flag = True
          if not flag:
              probs = probs_copy
          if self.select_type == 'greedy':

              _, selected = probs.max(dim=-1)
              selected = selected.unsqueeze(0)
          elif self.select_type == 'sampling':
              if isinstance(probs, torch.Tensor):

                  selected = probs.multinomial(1)
                  visitflag[selected] = True
              else:
                  selected = []
                  for item in probs:
                      selected.append(item.multinomial(1).squeeze(1))
                  visitflag[selected] = True

          return selected, visitflag

class Attention_(nn.Module):

    def __init__(self,n_embd=32,n_head=4):
        super(Attention_, self).__init__()

        self.n_embd=n_embd
        self.n_head=n_head
        self.key_=nn.Linear(n_embd,n_embd).to(device)
        self.query_=nn.Linear(n_embd,n_embd).to(device)
        self.value_=nn.Linear(n_embd,n_embd).to(device)

        self.proj=nn.Linear(n_embd,n_embd)
        self.proj2=nn.Linear(n_embd,n_embd)

    # batch_size,num_agents,graph_size,hidden_dim
    def forward(self,query,key,value):
        B,L,G,D=key.size()

        K=self.key_(key).view(B,L,self.n_head, G,D//self.n_head).transpose(1,2).transpose(2,3) # (B,nh,L,G,hdim) L=n_agents
        V = self.value_(value).view(B, L, self.n_head,G, D // self.n_head).transpose(1, 2).transpose(2,3) # (B,nh,G,L,hdim)
        Q = self.query_(query).view(B, L, self.n_head,G, D // self.n_head).transpose(1, 2).transpose(2,3)  # (B,nh,G,L,hdim)

        att=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(D)  # (B,nh,G,L,L)
        att=F.softmax(att,dim=-1)  # (B,nh,G,L,L)
        y=torch.matmul(att,V)      #  (B,nh,G,L,hdim)
        y=y.transpose(2,3).transpose(1,2).contiguous().view(B,L,G,D)
        y=F.relu(self.proj(y))
        y=self.proj2(y)
        return y

class SelfAttention_(nn.Module):

    def __init__(self,n_embd=32,n_head=4,norm_shape=32):
        super(SelfAttention_, self).__init__()

        self.n_embd=n_embd
        self.n_head=n_head
        self.key_=nn.Linear(n_embd,n_embd).to(device)
        self.query_=nn.Linear(n_embd,n_embd).to(device)
        self.value_=nn.Linear(n_embd,n_embd).to(device)
        self.ln=nn.LayerNorm(norm_shape)
        self.proj=nn.Linear(n_embd,n_embd)
        self.proj2 = nn.Linear(n_embd, n_embd)

    def forward(self,query,key,value): # (batch_size,num_agent,hidden_dim)
        B,L,D=key.size()

        K=self.ln(key+self.key_(key)).view(B,L,self.n_head, D//self.n_head).transpose(1,2) # (B,nh,L,hdim) L=n_agents
        V = self.ln(value+self.value_(value)).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B,nh,L,hdim)
        Q =self.ln(query +self.query_(query)).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B,nh,1,hdim)

        att=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(D)  # (B,nh,1,L)
        att=F.softmax(att,dim=-1)  # (B,nh,L,L)
        y=torch.matmul(att,V)      #  (B,nh,L,hdim)
        y=y.transpose(1,2).contiguous().view(B,L,D)
        y=F.relu(self.proj2(y))
        y=self.ln(self.proj(y))
        return y




