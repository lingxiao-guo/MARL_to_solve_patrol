import os
import sys
import argparse
from trainer.asy_ppo import asy_trainer
from trainer.sy_ppo import sy_trainer  #参数就通过这个往里边传，说明这个类要含有所有其他类的接口


parser=argparse.ArgumentParser(description = 'train')
parser.add_argument('--graph_size', type=int, default=20, help="size of the problem graph")
parser.add_argument('--lr', type=int, default=0.000025, help="learning rate")
parser.add_argument('--num_input', type=int, default=5, help="dimension of network input")
parser.add_argument('--num_hidden', type=int, default=32, help="number of hidden dim")
parser.add_argument('--num_layers', type=int, default=1, help="number of graph encoder blocks")
parser.add_argument('--num_head', type=int, default=4, help="number of heads used in multi-head attention")
parser.add_argument('--action_dim', type=int, default=2, help="dimension of output action")
parser.add_argument('--select_type', type=str, default='sampling', help="sampling / greedy")

parser.add_argument('--train_episodes', type=int, default=800, help="number of train epochs")
parser.add_argument('--gamma', type=int, default=0.98, help="discount of reward")
parser.add_argument('--len_episode', type=int, default=1000, help="length of one episode")
parser.add_argument('--policy_update_steps', type=int, default=5, help="number of update the policy network one time")
parser.add_argument('--batch_size', type=int, default=40, help="batch size")
parser.add_argument('--ent_coef', type=int, default=0.01, help="entropy loss coefficient")
parser.add_argument('--vf_coef', type=int, default=0.8, help="value loss coefficient")
parser.add_argument('--factor', type=int, default=0.3, help="scheduler decrease factor")
parser.add_argument('--patience', type=int, default=800, help="scheduler decrease patience")
parser.add_argument('--clip_coef', type=int, default=0.05, help="PPO clip factor")
parser.add_argument('--patrol_mode', type=str, default='asynchronous', help="The mode of multi-robots' patrol")

args=parser.parse_args()

if __name__ == "__main__":

    mode=args.patrol_mode

    if mode ==  "asynchronous":
        asy_tr=asy_trainer(args)
        asy_tr.train()
    else:
        sy_tr=sy_trainer(args)
        sy_tr.train()

    