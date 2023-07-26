import torch

from xuanpolicy.torch.policies import *
from xuanpolicy.torch.utils import *
from xuanpolicy.torch.representations import Basic_Identical
from .deterministic_marl import BasicQhead


class ActorNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(ActorNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], action_dim, None, None, None, device)[0])
        self.pi_logits = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.pi_logits(x)


class CriticNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)[:, :, 0]


class CentralizedCritic(nn.Module):
    def __init__(self,
                 state_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(CentralizedCritic, self).__init__()
        layers = []
        input_shape = (state_dim * n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], 1, None, None, None, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class COMA_CriticNet(nn.Module):
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 act_dim: int,
                 n_agents: int,
                 hidden_sizes: Sequence[int],
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(COMA_CriticNet, self).__init__()
        layers = []
        input_shape = (state_dim + obs_dim + act_dim * n_agents + n_agents,)
        for h in hidden_sizes:
            mlp, input_shape = mlp_block(input_shape[0], h, normalize, activation, initialize, device)
            layers.extend(mlp)
        layers.extend(mlp_block(input_shape[0], act_dim, None, None, None, device)[0])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class MAAC_Policy(nn.Module):
    """
    MAAC_Policy: Multi-Agent Actor-Critic Policy
    """
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 mixer: Optional[VDN_mixer] = None,
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 **kwargs):
        super(MAAC_Policy, self).__init__()
        self.device = device
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.representation = representation
        self.representation_critic = copy.deepcopy(self.representation)
        self.representation_info_shape = self.representation.output_shapes
        self.lstm = True if kwargs["rnn"] == "LSTM" else False
        self.use_rnn = True if kwargs["use_recurrent"] else False
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, activation, device)
        self.centralized_V = True if kwargs['use_centralized_V'] else False
        critic_net = CentralizedCritic if self.centralized_V else CriticNet
        self.critic = critic_net(representation.output_shapes['state'][0], n_agents, critic_hidden_size,
                                 normalize, initialize, activation, device)
        self.mixer = mixer
        self.pi_dist = CategoricalDistribution(self.action_dim)

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                *rnn_hidden: torch.Tensor, avail_actions=None):
        if self.use_rnn:
            outputs = self.representation(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation(observation)
            rnn_hidden = None
        actor_input = torch.concat([outputs['state'], agent_ids], dim=-1)
        act_logits = self.actor(actor_input)
        if avail_actions is not None:
            avail_actions = torch.Tensor(avail_actions)
            act_logits_detach = act_logits.clone().detach()
            act_logits_detach[avail_actions == 0] = -1e10
            self.pi_dist.set_param(logits=act_logits_detach)
        else:
            self.pi_dist.set_param(act_logits)

        return rnn_hidden, self.pi_dist

    def get_values(self, observation: torch.Tensor, agent_ids: torch.Tensor,
                   *rnn_hidden: torch.Tensor):
        shape_obs = observation.shape
        if self.use_rnn:
            if len(shape_obs) == 4:  # for critic training
                batch_size, n_agent, episode_length, dim_obs = shape_obs[0], shape_obs[1], shape_obs[2], shape_obs[3]
                outputs = self.representation_critic(observation.view(-1, episode_length, dim_obs), *rnn_hidden)
                outputs['state'] = outputs['state'].view(batch_size, n_agent, episode_length, -1)
            else:  # for interaction
                outputs = self.representation_critic(observation, *rnn_hidden)
            rnn_hidden = (outputs['rnn_hidden'], outputs['rnn_cell'])
        else:
            outputs = self.representation_critic(observation)
            rnn_hidden = None
        if self.centralized_V:  # use centralize critic with global features input
            if len(shape_obs) == 4:  # for critic training
                batch_size, n_agent, episode_length = shape_obs[0], shape_obs[1], shape_obs[2]
                critic_in = outputs['state'].transpose(1, 2).view(batch_size, episode_length, -1)
                v = self.critic(critic_in).unsqueeze(1).expand(-1, self.n_agents, -1, -1)
            else:  # agent-environment interaction
                batch_size = observation.shape[0]
                critic_in = outputs['state'].reshape(batch_size, -1)
                v = self.critic(critic_in).unsqueeze(1).expand(-1, self.n_agents, -1)
        else:
            if len(shape_obs) == 4:  # for critic training
                batch_size, n_agent, episode_length = shape_obs[0], shape_obs[1], shape_obs[2]
                critic_in = outputs['state'].transpose(1, 2).view(batch_size, episode_length, -1)
                critic_in = torch.concat([critic_in, agent_ids], dim=-1)
                v = self.critic(critic_in).unsqueeze(-1)
            else:  # agent-environment interaction
                critic_input = torch.concat([outputs['state'], agent_ids], dim=-1)
                v = self.critic(critic_input).unsqueeze(-1)
        return rnn_hidden, v

    def value_tot(self, values_n: torch.Tensor, global_state=None):
        if global_state is not None:
            global_state = torch.as_tensor(global_state).to(self.device)
        return values_n if self.mixer is None else self.mixer(values_n, global_state)


class MeanFieldActorCriticPolicy(nn.Module):
    def __init__(self,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None
                 ):
        super(MeanFieldActorCriticPolicy, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor_net = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                                  actor_hidden_size, normalize, initialize, activation, device)
        self.critic_net = BasicQhead(representation.output_shapes['state'][0] + self.action_dim, self.action_dim,
                                     n_agents, critic_hidden_size, normalize, initialize, activation, device)
        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)
        self.parameters_actor = list(self.actor_net.parameters()) + list(self.representation.parameters())
        self.parameters_critic = self.critic_net.parameters()

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        input_actor = torch.concat([outputs['state'], agent_ids], dim=-1)
        act_dist = self.actor_net(input_actor)
        return outputs, act_dist

    def target_actor(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        input_actor = torch.concat([outputs['state'], agent_ids], dim=-1)
        act_dist = self.target_actor_net(input_actor)
        return act_dist

    def critic(self, observation: torch.Tensor, actions_mean: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        critic_in = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
        return self.critic_net(critic_in)

    def target_critic(self, observation: torch.Tensor, actions_mean: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        critic_in = torch.concat([outputs['state'], actions_mean, agent_ids], dim=-1)
        return self.target_critic_net(critic_in)

    def soft_update(self, tau=0.005):
        for ep, tp in zip(self.actor_net.parameters(), self.target_actor_net.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)
        for ep, tp in zip(self.critic_net.parameters(), self.target_critic_net.parameters()):
            tp.data.mul_(1 - tau)
            tp.data.add_(tau * ep.data)


class COMAPolicy(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_space: Discrete,
                 n_agents: int,
                 representation: Optional[Basic_Identical],
                 actor_hidden_size: Sequence[int] = None,
                 critic_hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., torch.Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None):
        super(COMAPolicy, self).__init__()
        self.device = device
        self.action_dim = action_space.n
        self.n_agents = n_agents
        self.representation = representation
        self.representation_info_shape = self.representation.output_shapes
        self.actor = ActorNet(representation.output_shapes['state'][0], self.action_dim, n_agents,
                              actor_hidden_size, normalize, initialize, activation, device)
        self.critic = COMA_CriticNet(state_dim, representation.output_shapes['state'][0], self.action_dim, n_agents,
                                     critic_hidden_size, normalize, initialize, activation, device)
        self.target_critic = copy.deepcopy(self.critic)
        self.parameters_critic = self.critic.parameters()
        self.parameters_actor = list(self.representation.parameters()) + list(self.actor.parameters())

    def build_critic_in(self, state, observations, actions_onehot, agent_ids, t=None):
        bs, act_dim = state.shape[0], actions_onehot.shape[-1]
        step_len = state.shape[1] if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        obs_encode = self.representation(observations)['state']
        inputs = [state[:, ts], obs_encode[:, ts]]
        # counterfactual actions inputs
        actions_joint = actions_onehot[:, ts].view(bs, step_len, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - torch.eye(self.n_agents)).view(-1, 1).repeat(1, act_dim).view(self.n_agents, -1)
        agent_mask = agent_mask.unsqueeze(0).unsqueeze(0).to(self.device)
        inputs.append(actions_joint * agent_mask)
        inputs.append(agent_ids[:, ts])
        return torch.concat(inputs, dim=-1)

    def forward(self, observation: torch.Tensor, agent_ids: torch.Tensor):
        outputs = self.representation(observation)
        input_with_id = torch.concat([outputs['state'], agent_ids], dim=-1)
        act_dist = self.actor(input_with_id)
        return outputs, act_dist

    def copy_target(self):
        for ep, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(ep)
