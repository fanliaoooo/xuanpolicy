PDQN_Agent
=====================================

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class:: 
    xuance.torch.agent.policy_gradient.pdqn_agent.PDQN_Agent(config, envs, policy, optimizer, scheduler, device)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param policy: The policy that provides actions and values.
    :type policy: nn.Module
    :param optimizer: The optimizer that updates the parameters.
    :type optimizer: torch.optim.Optimizer
    :param scheduler: Implement the learning rate decay.
    :type scheduler: torch.optim.lr_scheduler._LRScheduler
    :param device: Choose CPU or GPU to train the model.
    :type device: str, int, torch.device

.. py:function:: 
    xuance.torch.agent.policy_gradient.pdqn_agent.PDQN_Agent._action(obs, egreedy)

    Calculate actions according to the observations.

    :param obs: The observation of current step.
    :type obs: numpy.ndarray
    :param egreedy: The epsilong greedy factor.
    :type egreedy: np.float
    :return: **disaction**, **conaction**, **con_actions** - Discrete actions, continuous actions, continuous actions.
    :rtype: np.ndarray, np.ndarray, np.ndarray

.. py:function:: 
    xuance.torch.agent.policy_gradient.pdqn_agent.PDQN_Agent.pad_action(disaction, conaction)

    :param disaction: The discrete actions.
    :type disaction: numpy.ndarray
    :param conaction: The continuous actions.
    :type conaction: numpy.ndarray
    :return: **(disaction, con_actions)**
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
  
.. py:function:: 
    xuance.torch.agent.policy_gradient.pdqn_agent.PDQN_Agent.train(train_steps)

    Train the PDQN agent.

    :param train_steps: The number of steps for training.
    :type train_steps: int

.. py:function:: 
    xuance.torch.agent.policy_gradient.pdqn_agent.PDQN_Agent.test(env_fn, test_episodes)
  
    Test the trained model.

    :param env_fn: The function of making environments.
    :param test_episodes: The number of testing episodes.
    :type test_episodes: int
    :return: **scores** - The accumulated scores of these episodes.
    :rtype: list

.. raw:: html

    <br><hr>

**TensorFlow:**


.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
    xuance.mindspore.agents.policy_gradient.pdqn_agent.PDQN_Agent(config, envs, policy, optimizer, scheduler)

    :param config: Provides hyper parameters.
    :type config: Namespace
    :param envs: The vectorized environments.
    :type envs: xuance.environments.vector_envs.vector_env.VecEnv
    :param policy: The policy that provides actions and values.
    :type policy: nn.Module
    :param optimizer: The optimizer that updates the parameters.
    :type optimizer: torch.optim.Optimizer
    :param scheduler: Implement the learning rate decay.
    :type scheduler: torch.optim.lr_scheduler._LRScheduler

.. py:function::
    xuance.mindspore.agents.policy_gradient.pdqn_agent.PDQN_Agent._action(obs)

    :param obs: xxxxxx.
    :type obs: xxxxxx
    :return: xxxxxx.
    :rtype: xxxxxx

.. py:function::
    xuance.mindspore.agents.policy_gradient.pdqn_agent.PDQN_Agent.pad_action(disaction, conaction)

    :param disaction: xxxxxx.
    :type disaction: xxxxxx
    :param conaction: xxxxxx.
    :type conaction: xxxxxx
    :return: xxxxxx.
    :rtype: xxxxxx

.. py:function::
    xuance.mindspore.agents.policy_gradient.pdqn_agent.PDQN_Agent.train(train_steps)

    :param train_steps: xxxxxx.
    :type train_steps: xxxxxx

.. py:function::
    xuance.mindspore.agents.policy_gradient.pdqn_agent.PDQN_Agent.test(env_fn,test_episodes)

    :param env_fn: xxxxxx.
    :type env_fn: xxxxxx
    :param test_episodes: xxxxxx.
    :type test_episodes: xxxxxx
    :return: xxxxxx.
    :rtype: xxxxxx

.. py:function::
    xuance.mindspore.agents.policy_gradient.pdqn_agent.PDQN_Agent.end_episode(episode)

    :param episode: xxxxxx.
    :type episode: xxxxxx

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::
  
    .. group-tab:: PyTorch
    
        .. code-block:: python

            import numpy as np

            from xuance.torch.agents import *
            import gym
            from gym import spaces

            class PDQN_Agent(Agent):
                def __init__(self,
                            config: Namespace,
                            envs: Gym_Env,
                            policy: nn.Module,
                            optimizer: Sequence[torch.optim.Optimizer],
                            scheduler: Optional[Sequence[torch.optim.lr_scheduler._LRScheduler]] = None,
                            device: Optional[Union[int, str, torch.device]] = None):
                    self.envs = envs
                    self.comm = MPI.COMM_WORLD
                    self.render = config.render

                    self.gamma = config.gamma
                    self.use_obsnorm = config.use_obsnorm
                    self.use_rewnorm = config.use_rewnorm
                    self.obsnorm_range = config.obsnorm_range
                    self.rewnorm_range = config.rewnorm_range

                    self.train_frequency = config.training_frequency
                    self.start_training = config.start_training
                    self.start_noise = config.start_noise
                    self.end_noise = config.end_noise
                    self.noise_scale = config.start_noise

                    self.observation_space = envs.observation_space.spaces[0]
                    old_as = envs.action_space
                    num_disact = old_as.spaces[0].n
                    self.action_space = gym.spaces.Tuple((old_as.spaces[0], *(gym.spaces.Box(old_as.spaces[1].spaces[i].low,
                                                    old_as.spaces[1].spaces[i].high, dtype=np.float32) for i in range(0, num_disact))))
                    self.action_high = [self.action_space.spaces[i].high for i in range(1, num_disact + 1)]
                    self.action_low = [self.action_space.spaces[i].low for i in range(1, num_disact + 1)]
                    self.action_range = [self.action_space.spaces[i].high - self.action_space.spaces[i].low for i in range(1, num_disact + 1)]
                    self.representation_info_shape = {'state': (envs.observation_space.spaces[0].shape)}
                    self.auxiliary_info_shape = {}
                    self.nenvs = 1
                    self.epsilon = 1.0
                    self.epsilon_steps = 1000
                    self.epsilon_initial = 1.0
                    self.epsilon_final = 0.1
                    self.buffer_action_space = spaces.Box(np.zeros(4), np.ones(4), dtype=np.float64)

                    memory = DummyOffPolicyBuffer(self.observation_space,
                                                self.buffer_action_space,
                                                self.representation_info_shape,
                                                self.auxiliary_info_shape,
                                                self.nenvs,
                                                config.nsize,
                                                config.batchsize)
                    learner = PDQN_Learner(policy,
                                        optimizer,
                                        scheduler,
                                        config.device,
                                        config.model_dir,
                                        config.gamma,
                                        config.tau)

                    self.num_disact = self.action_space.spaces[0].n
                    self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact+1)])
                    self.conact_size = int(self.conact_sizes.sum())

                    self.obs_rms = RunningMeanStd(shape=space2shape(self.observation_space), comm=self.comm, use_mpi=False)
                    self.ret_rms = RunningMeanStd(shape=(), comm=self.comm, use_mpi=False)
                    super(PDQN_Agent, self).__init__(envs, policy, memory, learner, device, config.log_dir, config.model_dir)

                def _process_observation(self, observations):
                    if self.use_obsnorm:
                        if isinstance(self.observation_space, gym.spaces.Dict):
                            for key in self.observation_space.spaces.keys():
                                observations[key] = np.clip(
                                    (observations[key] - self.obs_rms.mean[key]) / (self.obs_rms.std[key] + EPS),
                                    -self.obsnorm_range, self.obsnorm_range)
                        else:
                            observations = np.clip((observations - self.obs_rms.mean) / (self.obs_rms.std + EPS),
                                                -self.obsnorm_range, self.obsnorm_range)
                        return observations
                    return observations

                def _process_reward(self, rewards):
                    if self.use_rewnorm:
                        std = np.clip(self.ret_rms.std, 0.1, 100)
                        return np.clip(rewards / std, -self.rewnorm_range, self.rewnorm_range)
                    return rewards

                def _action(self, obs):
                    with torch.no_grad():
                        obs = torch.as_tensor(obs, device=self.device).float()
                        con_actions = self.policy.con_action(obs)
                        rnd = np.random.rand()
                        if rnd < self.epsilon:
                            disaction = np.random.choice(self.num_disact)
                        else:
                            q = self.policy.Qeval(obs.unsqueeze(0), con_actions.unsqueeze(0))
                            q = q.detach().cpu().data.numpy()
                            disaction = np.argmax(q)

                    con_actions = con_actions.cpu().data.numpy()
                    offset = np.array([self.conact_sizes[i] for i in range(disaction)], dtype=int).sum()
                    conaction = con_actions[offset:offset+self.conact_sizes[disaction]]

                    return disaction, conaction, con_actions

                def pad_action(self, disaction, conaction):
                    con_actions = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
                    con_actions[disaction][:] = conaction
                    return (disaction, con_actions)

                def train(self, train_steps=10000):
                    episodes = np.zeros((self.nenvs,), np.int32)
                    scores = np.zeros((self.nenvs,), np.float32)
                    returns = np.zeros((self.nenvs,), np.float32)
                    obs, _ = self.envs.reset()
                    for step in tqdm(range(train_steps)):
                        step_info, episode_info = {}, {}
                        disaction, conaction, con_actions = self._action(obs)
                        action = self.pad_action(disaction, conaction)
                        action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[disaction]
                        (next_obs, steps), rewards, terminal, _ = self.envs.step(action)
                        if self.render: self.envs.render("human")
                        acts = np.concatenate(([disaction], con_actions), axis=0).ravel()
                        state = {'state': obs}
                        self.memory.store(obs, acts, rewards, terminal, next_obs, state, {})
                        if step > self.start_training and step % self.train_frequency == 0:
                            obs_batch, act_batch, rew_batch, terminal_batch, next_batch, _, _ = self.memory.sample()
                            step_info = self.learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)
                        scores += rewards
                        returns = self.gamma * returns + rewards
                        obs = next_obs
                        self.noise_scale = self.start_noise - (self.start_noise - self.end_noise) / train_steps
                        if terminal == True:
                            step_info["returns-step"] = scores
                            episode_info["returns-episode"] = scores
                            scores = 0
                            returns = 0
                            episodes += 1
                            self.end_episode(episodes)
                            obs, _ = self.envs.reset()
                            self.log_infos(step_info, step)
                            self.log_infos(episode_info, episodes)
                        if step % 50000 == 0 or step == train_steps - 1:
                            self.save_model()
                            np.save(self.model_dir + "/obs_rms.npy",
                                    {'mean': self.obs_rms.mean, 'std': self.obs_rms.std, 'count': self.obs_rms.count})

                def end_episode(self, episode):
                    if episode < self.epsilon_steps:
                        self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                                episode / self.epsilon_steps)
                    else:
                        self.epsilon = self.epsilon_final

                def test(self, test_steps=10000, load_model=None):
                    self.load_model(self.model_dir)
                    scores = np.zeros((self.nenvs,), np.float32)
                    returns = np.zeros((self.nenvs,), np.float32)
                    obs, _ = self.envs.reset()
                    for _ in tqdm(range(test_steps)):
                        disaction, conaction, con_actions = self._action(obs)
                        action = self.pad_action(disaction, conaction)
                        action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[disaction]
                        (next_obs, steps), rewards, terminal, _ = self.envs.step(action)
                        self.envs.render("human")
                        scores += rewards
                        returns = self.gamma * returns + rewards
                        obs = next_obs
                        if terminal == True:
                            scores, returns = 0, 0
                            obs, _ = self.envs.reset()

                def evaluate(self):
                    pass



    .. group-tab:: TensorFlow
    
        .. code-block:: python3



    .. group-tab:: MindSpore

        .. code-block:: python

            from xuance.mindspore.agents import *
            import gym
            from gym import spaces

            class PDQN_Agent(Agent):
                def __init__(self,
                             config: Namespace,
                             envs: Gym_Env,
                             policy: nn.Cell,
                             optimizer: Sequence[nn.Optimizer],
                             scheduler):
                    self.envs = envs
                    self.render = config.render
                    self.n_envs = envs.num_envs

                    self.gamma = config.gamma
                    self.train_frequency = config.training_frequency
                    self.start_training = config.start_training
                    self.start_greedy = config.start_greedy
                    self.end_greedy = config.end_greedy
                    self.egreedy = config.start_greedy

                    self.train_frequency = config.training_frequency
                    self.start_training = config.start_training
                    self.start_noise = config.start_noise
                    self.end_noise = config.end_noise
                    self.noise_scale = config.start_noise

                    self.observation_space = envs.observation_space.spaces[0]
                    old_as = envs.action_space
                    num_disact = old_as.spaces[0].n
                    self.action_space = gym.spaces.Tuple((old_as.spaces[0], *(gym.spaces.Box(old_as.spaces[1].spaces[i].low,
                                                                                             old_as.spaces[1].spaces[i].high,
                                                                                             dtype=np.float32) for i in
                                                                              range(0, num_disact))))
                    self.action_high = [self.action_space.spaces[i].high for i in range(1, num_disact + 1)]
                    self.action_low = [self.action_space.spaces[i].low for i in range(1, num_disact + 1)]
                    self.action_range = [self.action_space.spaces[i].high - self.action_space.spaces[i].low for i in
                                         range(1, num_disact + 1)]
                    self.representation_info_shape = {'state': (envs.observation_space.spaces[0].shape)}
                    self.auxiliary_info_shape = {}
                    self.nenvs = 1
                    self.epsilon = 1.0
                    self.epsilon_steps = 1000
                    self.epsilon_initial = 1.0
                    self.epsilon_final = 0.1
                    self.buffer_action_space = spaces.Box(np.zeros(4), np.ones(4), dtype=np.float64)

                    memory = DummyOffPolicyBuffer(self.observation_space,
                                                  self.buffer_action_space,
                                                  self.auxiliary_info_shape,
                                                  self.n_envs,
                                                  config.n_size,
                                                  config.batch_size)
                    learner = PDQN_Learner(policy,
                                           optimizer,
                                           scheduler,
                                           config.model_dir,
                                           config.gamma,
                                           config.tau)

                    self.num_disact = self.action_space.spaces[0].n
                    self.conact_sizes = np.array([self.action_space.spaces[i].shape[0] for i in range(1, self.num_disact+1)])
                    self.conact_size = int(self.conact_sizes.sum())

                    super(PDQN_Agent, self).__init__(config, envs, policy, memory, learner, config.log_dir, config.model_dir)

                def _action(self, obs):
                    obs = ms.Tensor(obs)
                    con_actions = self.policy.con_action(obs)
                    rnd = np.random.rand()
                    if rnd < self.epsilon:
                        disaction = np.random.choice(self.num_disact)
                    else:
                        q = self.policy.Qeval(obs.expand_dims(0), con_actions.expand_dims(0))
                        q = q.asnumpy()
                        disaction = np.argmax(q)
                    con_actions = con_actions.asnumpy()
                    offset = np.array([self.conact_sizes[i] for i in range(disaction)], dtype=int).sum()
                    conaction = con_actions[offset:offset+self.conact_sizes[disaction]]

                    return disaction, conaction, con_actions

                def pad_action(self, disaction, conaction):
                    con_actions = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
                    con_actions[disaction][:] = conaction
                    return (disaction, con_actions)

                def train(self, train_steps=10000):
                    episodes = np.zeros((self.nenvs,), np.int32)
                    scores = np.zeros((self.nenvs,), np.float32)
                    obs, _ = self.envs.reset()
                    for _ in tqdm(range(train_steps)):
                        step_info = {}
                        disaction, conaction, con_actions = self._action(obs)
                        action = self.pad_action(disaction, conaction)
                        action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[
                            disaction]
                        (next_obs, steps), rewards, terminal, _ = self.envs.step(action)
                        if self.render: self.envs.render("human")
                        acts = np.concatenate(([disaction], con_actions), axis=0).ravel()
                        self.memory.store(obs, acts, rewards, terminal, next_obs)
                        if self.current_step > self.start_training and self.current_step % self.train_frequency == 0:
                            obs_batch, act_batch, rew_batch, terminal_batch, next_batch = self.memory.sample()
                            step_info = self.learner.update(obs_batch, act_batch, rew_batch, next_batch, terminal_batch)

                        scores += rewards
                        obs = next_obs
                        self.noise_scale = self.start_noise - (self.start_noise - self.end_noise) / train_steps
                        if terminal == True:
                            step_info["returns-step"] = scores
                            scores = 0
                            returns = 0
                            episodes += 1
                            self.end_episode(episodes)
                            obs, _ = self.envs.reset()
                            self.log_infos(step_info, self.current_step)

                        self.current_step += self.n_envs
                        if self.egreedy >= self.end_greedy:
                            self.egreedy = self.egreedy - (self.start_greedy - self.end_greedy) / self.config.decay_step_greedy

                def test(self, env_fn, test_episodes):
                    test_envs = env_fn()
                    episode_score = 0
                    current_episode, scores, best_score = 0, [], -np.inf
                    obs, _ = self.envs.reset()

                    while current_episode < test_episodes:
                        disaction, conaction, con_actions = self._action(obs)
                        action = self.pad_action(disaction, conaction)
                        action[1][disaction] = self.action_range[disaction] * (action[1][disaction] + 1) / 2. + self.action_low[
                            disaction]
                        (next_obs, steps), rewards, terminal, _ = self.envs.step(action)
                        self.envs.render("human")
                        episode_score += rewards
                        obs = next_obs
                        if terminal == True:
                            scores.append(episode_score)
                            obs, _ = self.envs.reset()
                            current_episode += 1
                            if best_score < episode_score:
                                best_score = episode_score
                            episode_score = 0
                            if self.config.test_mode:
                                print("Episode: %d, Score: %.2f" % (current_episode, episode_score))

                    if self.config.test_mode:
                        print("Best Score: %.2f" % (best_score))

                    test_info = {
                        "Test-Episode-Rewards/Mean-Score": np.mean(scores),
                        "Test-Episode-Rewards/Std-Score": np.std(scores)
                    }
                    self.log_infos(test_info, self.current_step)

                    test_envs.close()

                    return scores

                def end_episode(self, episode):
                    if episode < self.epsilon_steps:
                        self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                                episode / self.epsilon_steps)
                    else:
                        self.epsilon = self.epsilon_final
