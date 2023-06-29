import torch.nn

from xuanpolicy.torch.agents import *
from xuanpolicy.torch.agents.agents_marl import linear_decay_or_increase


class DCG_Agents(MARLAgents):
    def __init__(self,
                 config: Namespace,
                 envs: DummyVecEnv_MAS,
                 device: Optional[Union[int, str, torch.device]] = None):
        self.gamma = config.gamma
        self.start_greedy = config.start_greedy
        self.end_greedy = config.end_greedy
        self.egreedy = config.start_greedy

        input_representation = get_repre_in(config)
        representation = REGISTRY_Representation[config.representation](*input_representation)
        repre_state_dim = config.representation_hidden_size[-1]
        from xuanpolicy.torch.policies.coordination_graph import DCG_utility, DCG_payoff, Coordination_Graph
        utility = DCG_utility(repre_state_dim, config.hidden_utility_dim, config.dim_act).to(device)
        payoffs = DCG_payoff(repre_state_dim * 2, config.hidden_payoff_dim, config.dim_act, config).to(device)
        dcgraph = Coordination_Graph(config.n_agents, config.graph_type)
        dcgraph.set_coordination_graph(device)
        if config.agent == "DCG_S":
            policy = REGISTRY_Policy[config.policy](config.action_space[config.agent_keys[0]],
                                                    config.state_space.shape[0], representation,
                                                    utility, payoffs, dcgraph, config.hidden_bias_dim,
                                                    None, None, torch.nn.ReLU, device)
        else:
            policy = REGISTRY_Policy[config.policy](config.action_space[config.agent_keys[0]],
                                                    config.state_space.shape[0], representation,
                                                    utility, payoffs, dcgraph, None,
                                                    None, None, torch.nn.ReLU, device)
        optimizer = torch.optim.Adam(policy.parameters(), config.learning_rate, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5,
                                                      total_iters=get_total_iters(config.agent_name, config))
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        self.representation_info_shape = policy.representation.output_shapes
        self.auxiliary_info_shape = {}

        if config.state_space is not None:
            config.dim_state, state_shape = config.state_space.shape, config.state_space.shape
        else:
            config.dim_state, state_shape = None, None
        memory = MARL_OffPolicyBuffer(state_shape,
                                      config.obs_shape,
                                      config.act_shape,
                                      config.rew_shape,
                                      config.done_shape,
                                      envs.num_envs,
                                      config.buffer_size,
                                      config.batch_size)
        from xuanpolicy.torch.learners.multi_agent_rl.dcg_learner import DCG_Learner
        learner = DCG_Learner(config, policy, optimizer, scheduler,
                              config.device, config.modeldir, config.gamma,
                              config.sync_frequency)

        self.epsilon_decay = linear_decay_or_increase(config.start_greedy, config.end_greedy,
                                                      config.greedy_update_steps)
        super(DCG_Agents, self).__init__(config, envs, policy, memory, learner, device,
                                         config.logdir, config.modeldir)

    def act(self, obs_n, episode=None, test_mode=True, noise=False):
        return self.learner.act(obs_n, episode, test_mode, noise)

    def train(self, i_episode):
        self.epsilon_decay.update()
        if self.memory.can_sample(self.args.batch_size):
            sample = self.memory.sample()
            info_train = self.learner.update(sample)
            return info_train
        else:
            return {}