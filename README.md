# MARL_to_solve_patrol

Patrolling with multiple robots is a challenging task. While the robots collaboratively and repeatedly cover the regions of interest in the environment, their routes should satisfy two often conflicting properties: i) (efficiency) the times between two consecutive visits to the regions are small; ii) (unpredictability) the patrolling trajectories are random and unpredictable. We manage to strike a balance between the two goals by i) recasting the original patrolling problem as a Graph Deep Learning problem; ii) directly solving this problem on the graph in the framework of cooperative multi-agent reinforcement learning. Treating the movement of a team of agents as a sequence input, our model outputs the agents' actions in order by an autoregressive mechanism. Extensive simulation studies show that our approach has comparable performance with SOTA algorithms in terms of efficiency and outperforms them in terms of unpredictability.

## Paper
The paper is submitted to ICRA2022.
