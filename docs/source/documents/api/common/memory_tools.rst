Memory
==============================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:function::
  xuance.common.memory_tools.create_memory(shape, n_envs, n_size, dtype)

  xxxxxx.

  :param shape: xxxxxx.
  :type shape: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param n_size: xxxxxx.
  :type n_size: xxxxxx
  :param dtype: xxxxxx.
  :type dtype: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.common.memory_tools.store_element(data, memory, ptr)

  xxxxxx.

  :param data: xxxxxx.
  :type data: xxxxxx
  :param memory: xxxxxx.
  :type memory: xxxxxx
  :param ptr: xxxxxx.
  :type ptr: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.common.memory_tools.sample_batch(memory, index)

  xxxxxx.

  :param memory: xxxxxx.
  :type memory: xxxxxx
  :param index: xxxxxx.
  :type index: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.common.memory_tools.Buffer(observation_space, action_space, auxiliary_info_shape)

  :param observation_space: xxxxxx.
  :type observation_space: xxxxxx
  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param auxiliary_info_shape: xxxxxx.
  :type auxiliary_info_shape: xxxxxx

.. py:function::
  xuance.common.memory_tools.Buffer.full()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools.Buffer.store(*args)

  xxxxxx.

  :param *args: xxxxxx.
  :type *args: xxxxxx

.. py:function::
  xuance.common.memory_tools.Buffer.clear(*args)

  xxxxxx.

  :param *args: xxxxxx.
  :type *args: xxxxxx

.. py:function::
  xuance.common.memory_tools.Buffer.sample(*args)

  xxxxxx.

  :param *args: xxxxxx.
  :type *args: xxxxxx

.. py:function::
  xuance.common.memory_tools.Buffer.finish_path(*args)

  xxxxxx.

  :param *args: xxxxxx.
  :type *args: xxxxxx

.. py:class::
  xuance.common.memory_tools.EpisodeBuffer(obs, action, reward, done)

  :param obs: xxxxxx.
  :type obs: xxxxxx
  :param action: xxxxxx.
  :type action: xxxxxx
  :param reward: xxxxxx.
  :type reward: xxxxxx
  :param done: xxxxxx.
  :type done: xxxxxx

.. py:function::
  xuance.common.memory_tools.EpisodeBuffer.put(transition)

  xxxxxx.

  :param transition: xxxxxx.
  :type transition: xxxxxx

.. py:function::
  xuance.common.memory_tools.EpisodeBuffer.sample(lookup_step, idx)

  xxxxxx.

  :param lookup_step: xxxxxx.
  :type lookup_step: xxxxxx
  :param idx: xxxxxx.
  :type idx: xxxxxx
  :return: xxxxxx.
  :rtype: Dict[str, np.ndarray]

.. py:function::
  xuance.common.memory_tools.EpisodeBuffer.__len__(lookup_step, idx)

  xxxxxx.

  :return: xxxxxx.
  :rtype: int

.. py:class::
  xuance.common.memory_tools.DummyOnPolicyBuffer(observation_space, action_space, auxiliary_shape, n_envs, n_size, use_gae, use_advnorm, gamma, gae_lam)

  :param observation_space: xxxxxx.
  :type observation_space: xxxxxx
  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param auxiliary_shape: xxxxxx.
  :type auxiliary_shape: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param n_size: xxxxxx.
  :type n_size: xxxxxx
  :param use_gae: xxxxxx.
  :type use_gae: xxxxxx
  :param use_advnorm: xxxxxx.
  :type use_advnorm: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx
  :param gae_lam: xxxxxx.
  :type gae_lam: xxxxxx

.. py:function::
  xuance.common.memory_tools.DummyOnPolicyBuffer.full()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools.DummyOnPolicyBuffer.clear()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools.DummyOnPolicyBuffer.store(obs, acts, rews, value, terminals, aux_info)

  xxxxxx.

  :param obs: xxxxxx.
  :type obs: xxxxxx
  :param acts: xxxxxx.
  :type acts: xxxxxx
  :param rews: xxxxxx.
  :type rews: xxxxxx
  :param value: xxxxxx.
  :type value: xxxxxx
  :param terminals: xxxxxx.
  :type terminals: xxxxxx
  :param aux_info: xxxxxx.
  :type aux_info: xxxxxx

.. py:function::
  xuance.common.memory_tools.DummyOnPolicyBuffer.finish_path(val, i)

  xxxxxx.

  :param val: xxxxxx.
  :type val: xxxxxx
  :param i: xxxxxx.
  :type i: xxxxxx

.. py:function::
  xuance.common.memory_tools.DummyOnPolicyBuffer.sample(indexes)

  xxxxxx.

  :param indexes: xxxxxx.
  :type indexes: xxxxxx

.. py:class::
  xuance.common.memory_tools.DummyOffPolicyBuffer(observation_space, action_space, auxiliary_shape, n_envs, n_size, batch_size)

  :param observation_space: xxxxxx.
  :type observation_space: xxxxxx
  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param auxiliary_shape: xxxxxx.
  :type auxiliary_shape: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param n_size: xxxxxx.
  :type n_size: xxxxxx
  :param batch_size: xxxxxx.
  :type batch_size: xxxxxx

.. py:function::
  xuance.common.memory_tools.DummyOffPolicyBuffer.clear()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools.DummyOffPolicyBuffer.store(obs, acts, rews, terminals, next_obs)

  xxxxxx.

  :param obs: xxxxxx.
  :type obs: xxxxxx
  :param acts: xxxxxx.
  :type acts: xxxxxx
  :param rews: xxxxxx.
  :type rews: xxxxxx
  :param terminals: xxxxxx.
  :type terminals: xxxxxx
  :param next_obs: xxxxxx.
  :type next_obs: xxxxxx

.. py:function::
  xuance.common.memory_tools.DummyOffPolicyBuffer.sample(indexes)

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.common.memory_tools.RecurrentOffPolicyBuffer(observation_space, action_space, auxiliary_shape, n_envs, n_size, batch_size, episode_length, lookup_length)

  :param observation_space: xxxxxx.
  :type observation_space: xxxxxx
  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param auxiliary_shape: xxxxxx.
  :type auxiliary_shape: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param n_size: xxxxxx.
  :type n_size: xxxxxx
  :param batch_size: xxxxxx.
  :type batch_size: xxxxxx
  :param episode_length: xxxxxx.
  :type episode_length: xxxxxx
  :param lookup_length: xxxxxx.
  :type lookup_length: xxxxxx

.. py:function::
  xuance.common.memory_tools.RecurrentOffPolicyBuffer.full()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools.RecurrentOffPolicyBuffer.clear(*args)

  xxxxxx.
  :param *args: xxxxxx.
  :type *args: xxxxxx

.. py:function::
  xuance.common.memory_tools.RecurrentOffPolicyBuffer.store(episode)

  xxxxxx.

  :param episode: xxxxxx.
  :type episode: xxxxxx

.. py:function::
  xuance.common.memory_tools.RecurrentOffPolicyBuffer.sample()

  xxxxxx.

  :return: xxxxxx.
  :rtype: xxxxxx

.. py:class::
  xuance.common.memory_tools.PerOffPolicyBuffer(observation_space, action_space, auxiliary_shape, n_envs, n_size, batch_size, alpha)

  :param observation_space: xxxxxx.
  :type observation_space: xxxxxx
  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param auxiliary_shape: xxxxxx.
  :type auxiliary_shape: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param n_size: xxxxxx.
  :type n_size: xxxxxx
  :param batch_size: xxxxxx.
  :type batch_size: xxxxxx
  :param alpha: xxxxxx.
  :type alpha: xxxxxx

.. py:function::
  xuance.common.memory_tools.PerOffPolicyBuffer._sample_proportional(env_idx, batch_size)

  xxxxxx.

  :param env_idx: xxxxxx.
  :type env_idx: xxxxxx
  :param batch_size: xxxxxx.
  :type batch_size: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.common.memory_tools.PerOffPolicyBuffer.clear()

  xxxxxx.

.. py:function::
  xuance.common.memory_tools.PerOffPolicyBuffer.store(obs, acts, rews, terminals, next_obs)

  xxxxxx.

  :param obs: xxxxxx.
  :type obs: xxxxxx
  :param acts: xxxxxx.
  :type acts: xxxxxx
  :param rews: xxxxxx.
  :type rews: xxxxxx
  :param terminals: xxxxxx.
  :type terminals: xxxxxx
  :param next_obs: xxxxxx.
  :type next_obs: xxxxxx

.. py:function::
  xuance.common.memory_tools.PerOffPolicyBuffer.sample(beta)

  xxxxxx.

  :param beta: xxxxxx.
  :type beta: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.common.memory_tools.PerOffPolicyBuffer.update_priorities(idxes, priorities)

  xxxxxx.

  :param idxes: xxxxxx.
  :type idxes: xxxxxx
  :param priorities: xxxxxx.
  :type priorities: xxxxxx

.. py:class::
  xuance.common.memory_tools.DummyOffPolicyBuffer_Atari(observation_space, action_space, auxiliary_shape, n_envs, n_size, batch_size)

  :param observation_space: xxxxxx.
  :type observation_space: xxxxxx
  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param auxiliary_shape: xxxxxx.
  :type auxiliary_shape: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param n_size: xxxxxx.
  :type n_size: xxxxxx
  :param batch_size: xxxxxx.
  :type batch_size: xxxxxx

.. py:function::
  xuance.common.memory_tools.DummyOffPolicyBuffer_Atari.clear()

  xxxxxx.

.. py:class::
  xuance.common.memory_tools.DummyOffPolicyBuffer_Atari(observation_space, action_space, auxiliary_shape, n_envs, n_size, use_gae, use_advnorm, gamma, gae_lam)

  :param observation_space: xxxxxx.
  :type observation_space: xxxxxx
  :param action_space: xxxxxx.
  :type action_space: xxxxxx
  :param auxiliary_shape: xxxxxx.
  :type auxiliary_shape: xxxxxx
  :param n_envs: xxxxxx.
  :type n_envs: xxxxxx
  :param n_size: xxxxxx.
  :type n_size: xxxxxx
  :param use_gae: xxxxxx.
  :type use_gae: xxxxxx
  :param use_advnorm: xxxxxx.
  :type use_advnorm: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx
  :param gae_lam: xxxxxx.
  :type gae_lam: xxxxxx

.. py:function::
  xuance.common.memory_tools.DummyOffPolicyBuffer_Atari.clear()

  xxxxxx.

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        import random
        import numpy as np
        from gym import Space
        from abc import ABC, abstractmethod
        from typing import Optional, Union
        from xuance.common import space2shape, discount_cumsum
        from xuance.common.segtree_tool import SumSegmentTree, MinSegmentTree
        from collections import deque
        from typing import Dict


        def create_memory(shape: Optional[Union[tuple, dict]],
                          n_envs: int,
                          n_size: int,
                          dtype: type = np.float32):
            """
            Create a numpy array for memory data.
                shape: data shape.
                n_envs: number of parallel environments.
                n_size: length of data sequence for each environment.
                dtype: numpy data type.
            """
            if shape is None:
                return None
            elif isinstance(shape, dict):
                memory = {}
                for key, value in zip(shape.keys(), shape.values()):
                    if value is None:  # save an object type
                        memory[key] = np.zeros([n_envs, n_size], dtype=object)
                    else:
                        memory[key] = np.zeros([n_envs, n_size] + list(value), dtype=dtype)
                return memory
            elif isinstance(shape, tuple):
                return np.zeros([n_envs, n_size] + list(shape), dtype)
            else:
                raise NotImplementedError


        def store_element(data: Optional[Union[np.ndarray, dict, float]],
                          memory: Union[dict, np.ndarray],
                          ptr: int):
            """
            Insert a step of data into current memory.
                data: target data that to be stored.
                memory: the memory where data will be stored.
                ptr: pointer to the location for the data.
            """
            if data is None:
                return
            elif isinstance(data, dict):
                for key, value in zip(data.keys(), data.values()):
                    memory[key][:, ptr] = data[key]
            else:
                memory[:, ptr] = data


        def sample_batch(memory: Optional[Union[np.ndarray, dict]],
                         index: Optional[Union[np.ndarray, tuple]]):
            """
            Sample a batch of data from the selected memory.
                memory: memory that contains experience data.
                index: pointer to the location for the selected data.
            """
            if memory is None:
                return None
            elif isinstance(memory, dict):
                batch = {}
                for key, value in zip(memory.keys(), memory.values()):
                    batch[key] = value[index]
                return batch
            else:
                return memory[index]


        class Buffer(ABC):
            """
            Basic buffer single-agent DRL algorithms.
            """
            def __init__(self,
                         observation_space: Space,
                         action_space: Space,
                         auxiliary_info_shape: Optional[dict]):
                self.observation_space = observation_space
                self.action_space = action_space
                self.auxiliary_shape = auxiliary_info_shape
                self.size, self.ptr = 0, 0

            def full(self):
                pass

            @abstractmethod
            def store(self, *args):
                raise NotImplementedError

            @abstractmethod
            def clear(self, *args):
                raise NotImplementedError

            @abstractmethod
            def sample(self, *args):
                raise NotImplementedError

            def finish_path(self, *args):
                pass


        class EpisodeBuffer:
            """
            Episode buffer for DRQN agent.
            """
            def __init__(self):
                self.obs = []
                self.action = []
                self.reward = []
                self.done = []

            def put(self, transition):
                self.obs.append(transition[0])
                self.action.append(transition[1])
                self.reward.append(transition[2])
                self.done.append(transition[3])

            def sample(self, lookup_step=None, idx=None) -> Dict[str, np.ndarray]:
                obs = np.array(self.obs)
                action = np.array(self.action)
                reward = np.array(self.reward)
                done = np.array(self.done)

                obs = obs[idx:idx + lookup_step + 1]
                action = action[idx:idx + lookup_step]
                reward = reward[idx:idx + lookup_step]
                done = done[idx:idx + lookup_step]

                return dict(obs=obs,
                            acts=action,
                            rews=reward,
                            done=done)

            def __len__(self) -> int:
                return len(self.action)


        class DummyOnPolicyBuffer(Buffer):
            """
            Replay buffer for on-policy DRL algorithms.
                observation_space: the observation space of the environment.
                action_space: the action space of the environment.
                auxiliary_shape: data shape of auxiliary information (if exists).
                n_envs: number of parallel environments.
                n_size: max length of steps to store for one environment.
                use_gae: if use GAE trick.
                use_advnorm: if use Advantage normalization trick.
                gamma: discount factor.
                gae_lam: gae lambda.
            """
            def __init__(self,
                         observation_space: Space,
                         action_space: Space,
                         auxiliary_shape: Optional[dict],
                         n_envs: int,
                         n_size: int,
                         use_gae: bool = True,
                         use_advnorm: bool = True,
                         gamma: float = 0.99,
                         gae_lam: float = 0.95):
                super(DummyOnPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape)
                self.n_envs, self.n_size = n_envs, n_size
                self.buffer_size = self.n_size * self.n_envs
                self.use_gae, self.use_advnorm = use_gae, use_advnorm
                self.gamma, self.gae_lam = gamma, gae_lam
                self.start_ids = np.zeros(self.n_envs, np.int64)
                self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
                self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
                self.rewards = create_memory((), self.n_envs, self.n_size)
                self.returns = create_memory((), self.n_envs, self.n_size)
                self.values = create_memory((), self.n_envs, self.n_size)
                self.terminals = create_memory((), self.n_envs, self.n_size)
                self.advantages = create_memory((), self.n_envs, self.n_size)
                self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)

            @property
            def full(self):
                return self.size >= self.n_size

            def clear(self):
                self.ptr, self.size = 0, 0
                self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
                self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
                self.rewards = create_memory((), self.n_envs, self.n_size)
                self.returns = create_memory((), self.n_envs, self.n_size)
                self.values = create_memory((), self.n_envs, self.n_size)
                self.terminals = create_memory((), self.n_envs, self.n_size)
                self.advantages = create_memory((), self.n_envs, self.n_size)
                self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)

            def store(self, obs, acts, rews, value, terminals, aux_info=None):
                store_element(obs, self.observations, self.ptr)
                store_element(acts, self.actions, self.ptr)
                store_element(rews, self.rewards, self.ptr)
                store_element(value, self.values, self.ptr)
                store_element(terminals, self.terminals, self.ptr)
                store_element(aux_info, self.auxiliary_infos, self.ptr)
                self.ptr = (self.ptr + 1) % self.n_size
                self.size = min(self.size + 1, self.n_size)

            def finish_path(self, val, i):
                if self.full:
                    path_slice = np.arange(self.start_ids[i], self.n_size).astype(np.int32)
                else:
                    path_slice = np.arange(self.start_ids[i], self.ptr).astype(np.int32)
                vs = np.append(np.array(self.values[i, path_slice]), [val], axis=0)
                if self.use_gae:  # use gae
                    rewards = np.array(self.rewards[i, path_slice])
                    advantages = np.zeros_like(rewards)
                    dones = np.array(self.terminals[i, path_slice])
                    last_gae_lam = 0
                    step_nums = len(path_slice)
                    for t in reversed(range(step_nums)):
                        delta = rewards[t] + (1 - dones[t]) * self.gamma * vs[t + 1] - vs[t]
                        advantages[t] = last_gae_lam = delta + (1 - dones[t]) * self.gamma * self.gae_lam * last_gae_lam
                    returns = advantages + vs[:-1]
                else:
                    rewards = np.append(np.array(self.rewards[i, path_slice]), [val], axis=0)
                    returns = discount_cumsum(rewards, self.gamma)[:-1]
                    advantages = rewards[:-1] + self.gamma * vs[1:] - vs[:-1]

                self.returns[i, path_slice] = returns
                self.advantages[i, path_slice] = advantages
                self.start_ids[i] = self.ptr

            def sample(self, indexes):
                assert self.full, "Not enough transitions for on-policy buffer to random sample"

                env_choices, step_choices = divmod(indexes, self.n_size)

                obs_batch = sample_batch(self.observations, tuple([env_choices, step_choices]))
                act_batch = sample_batch(self.actions, tuple([env_choices, step_choices]))
                ret_batch = sample_batch(self.returns, tuple([env_choices, step_choices]))
                val_batch = sample_batch(self.values, tuple([env_choices, step_choices]))
                adv_batch = sample_batch(self.advantages, tuple([env_choices, step_choices]))
                if self.use_advnorm:
                    adv_batch = (adv_batch - np.mean(adv_batch)) / (np.std(adv_batch) + 1e-8)
                aux_batch = sample_batch(self.auxiliary_infos, tuple([env_choices, step_choices]))

                return obs_batch, act_batch, ret_batch, val_batch, adv_batch, aux_batch


        class DummyOffPolicyBuffer(Buffer):
            """
            Replay buffer for off-policy DRL algorithms.
                observation_space: the observation space of the environment.
                action_space: the action space of the environment.
                auxiliary_shape: data shape of auxiliary information (if exists).
                n_envs: number of parallel environments.
                n_size: max length of steps to store for one environment.
                batch_size: batch size of transition data for a sample.
            """
            def __init__(self,
                         observation_space: Space,
                         action_space: Space,
                         auxiliary_shape: Optional[dict],
                         n_envs: int,
                         n_size: int,
                         batch_size: int):
                super(DummyOffPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape)
                self.n_envs, self.n_size, self.batch_size = n_envs, n_size, batch_size
                self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
                self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
                self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
                self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)
                self.rewards = create_memory((), self.n_envs, self.n_size)
                self.terminals = create_memory((), self.n_envs, self.n_size)

            def clear(self):
                self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
                self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
                self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
                self.rewards = create_memory((), self.n_envs, self.n_size)
                self.terminals = create_memory((), self.n_envs, self.n_size)

            def store(self, obs, acts, rews, terminals, next_obs):
                store_element(obs, self.observations, self.ptr)
                store_element(acts, self.actions, self.ptr)
                store_element(rews, self.rewards, self.ptr)
                store_element(terminals, self.terminals, self.ptr)
                store_element(next_obs, self.next_observations, self.ptr)
                self.ptr = (self.ptr + 1) % self.n_size
                self.size = min(self.size + 1, self.n_size)

            def sample(self):
                env_choices = np.random.choice(self.n_envs, self.batch_size)
                step_choices = np.random.choice(self.size, self.batch_size)
                obs_batch = sample_batch(self.observations, tuple([env_choices, step_choices]))
                act_batch = sample_batch(self.actions, tuple([env_choices, step_choices]))
                rew_batch = sample_batch(self.rewards, tuple([env_choices, step_choices]))
                terminal_batch = sample_batch(self.terminals, tuple([env_choices, step_choices]))
                next_batch = sample_batch(self.next_observations, tuple([env_choices, step_choices]))
                return obs_batch, act_batch, rew_batch, terminal_batch, next_batch


        class RecurrentOffPolicyBuffer(Buffer):
            """
            Replay buffer for DRQN-based algorithms.
                observation_space: the observation space of the environment.
                action_space: the action space of the environment.
                auxiliary_shape: data shape of auxiliary information (if exists).
                n_envs: number of parallel environments.
                n_size: max length of steps to store for one environment.
                batch_size: batch size of transition data for a sample.
                episode_length: data length for an episode.
                lookup_length: the length of history data.
            """
            def __init__(self,
                         observation_space: Space,
                         action_space: Space,
                         auxiliary_shape: Optional[dict],
                         n_envs: int,
                         n_size: int,
                         batch_size: int,
                         episode_length: int,
                         lookup_length: int):
                super(RecurrentOffPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape)
                self.n_envs, self.n_size, self.episode_length, self.batch_size = n_envs, n_size, episode_length, batch_size
                self.lookup_length = lookup_length
                self.memory = deque(maxlen=self.n_size)

            @property
            def full(self):
                return self.size >= self.n_size

            def clear(self, *args):
                self.memory = deque(maxlen=self.n_size)

            def store(self, episode):
                self.memory.append(episode)
                self.ptr = (self.ptr + 1) % self.n_size
                self.size = min(self.size + 1, self.n_size)

            def sample(self):
                obs_batch, act_batch, rew_batch, terminal_batch = [], [], [], []
                episode_choices = np.random.choice(self.memory, self.batch_size)
                length_min = self.episode_length
                for episode in episode_choices:
                    length_min = min(length_min, len(episode))

                if length_min > self.lookup_length:
                    for episode in episode_choices:
                        start_idx = np.random.randint(0, len(episode) - self.lookup_length + 1)
                        sampled_data = episode.sample(lookup_step=self.lookup_length, idx=start_idx)
                        obs_batch.append(sampled_data["obs"])
                        act_batch.append(sampled_data["acts"])
                        rew_batch.append(sampled_data["rews"])
                        terminal_batch.append(sampled_data["done"])
                else:
                    for episode in episode_choices:
                        start_idx = np.random.randint(0, len(episode) - length_min + 1)
                        sampled_data = episode.sample(lookup_step=length_min, idx=start_idx)
                        obs_batch.append(sampled_data["obs"])
                        act_batch.append(sampled_data["acts"])
                        rew_batch.append(sampled_data["rews"])
                        terminal_batch.append(sampled_data["done"])

                return np.array(obs_batch), np.array(act_batch), np.array(rew_batch), np.array(terminal_batch)


        class PerOffPolicyBuffer(Buffer):
            """
            Prioritized Replay Buffer.
                observation_space: the observation space of the environment.
                action_space: the action space of the environment.
                auxiliary_shape: data shape of auxiliary information (if exists).
                n_envs: number of parallel environments.
                n_size: max length of steps to store for one environment.
                batch_size: batch size of transition data for a sample.
                alpha: prioritized factor.
            """
            def __init__(self,
                         observation_space: Space,
                         action_space: Space,
                         auxiliary_shape: Optional[dict],
                         n_envs: int,
                         n_size: int,
                         batch_size: int,
                         alpha: float = 0.6):
                super(PerOffPolicyBuffer, self).__init__(observation_space, action_space, auxiliary_shape)
                self.n_envs, self.n_size, self.batch_size = n_envs, n_size, batch_size
                self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
                self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
                self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
                self.rewards = create_memory((), self.n_envs, self.n_size)
                self.terminals = create_memory((), self.n_envs, self.n_size)

                self._alpha = alpha

                # set segment tree size
                it_capacity = 1
                while it_capacity < self.n_size:
                    it_capacity *= 2

                # init segment tree
                self._it_sum = []
                self._it_min = []
                for _ in range(n_envs):
                    self._it_sum.append(SumSegmentTree(it_capacity))
                    self._it_min.append(MinSegmentTree(it_capacity))
                self._max_priority = np.ones((n_envs))

            def _sample_proportional(self, env_idx, batch_size):
                res = []
                p_total = self._it_sum[env_idx].sum(0, self.size - 1)
                every_range_len = p_total / batch_size
                for i in range(batch_size):
                    mass = random.random() * every_range_len + i * every_range_len
                    idx = self._it_sum[env_idx].find_prefixsum_idx(mass)
                    res.append(int(idx))
                return res

            def clear(self):
                self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
                self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size)
                self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
                self.rewards = create_memory((), self.n_envs, self.n_size)
                self.terminals = create_memory((), self.n_envs, self.n_size)
                self._it_sum = []
                self._it_min = []

            def store(self, obs, acts, rews, terminals, next_obs):
                store_element(obs, self.observations, self.ptr)
                store_element(acts, self.actions, self.ptr)
                store_element(rews, self.rewards, self.ptr)
                store_element(terminals, self.terminals, self.ptr)
                store_element(next_obs, self.next_observations, self.ptr)

                # prioritized process
                for i in range(self.n_envs):
                    self._it_sum[i][self.ptr] = self._max_priority[i] ** self._alpha
                    self._it_min[i][self.ptr] = self._max_priority[i] ** self._alpha

                self.ptr = (self.ptr + 1) % self.n_size
                self.size = min(self.size + 1, self.n_size)

            def sample(self, beta):
                env_choices = np.array(range(self.n_envs)).repeat(int(self.batch_size / self.n_envs))
                step_choices = np.zeros((self.n_envs, int(self.batch_size / self.n_envs)))
                weights = np.zeros((self.n_envs, int(self.batch_size / self.n_envs)))

                assert beta > 0

                for i in range(self.n_envs):
                    idxes = self._sample_proportional(i, int(self.batch_size / self.n_envs))

                    weights_ = []
                    p_min = self._it_min[i].min() / self._it_sum[i].sum()
                    max_weight = p_min * self.size ** (-beta)

                    for idx in idxes:
                        p_sample = self._it_sum[i][idx] / self._it_sum[i].sum()
                        weight = p_sample * self.size ** (-beta)
                        weights_.append(weight / max_weight)
                    step_choices[i] = idxes
                    weights[i] = np.array(weights_)
                step_choices = step_choices.astype(np.uint8)

                obs_batch = sample_batch(self.observations, tuple([env_choices, step_choices.flatten()]))
                act_batch = sample_batch(self.actions, tuple([env_choices, step_choices.flatten()]))
                rew_batch = sample_batch(self.rewards, tuple([env_choices, step_choices.flatten()]))
                terminal_batch = sample_batch(self.terminals, tuple([env_choices, step_choices.flatten()]))
                next_batch = sample_batch(self.next_observations, tuple([env_choices, step_choices.flatten()]))

                # return tuple(list(encoded_sample) + [weights, idxes])
                return (obs_batch,
                        act_batch,
                        rew_batch,
                        terminal_batch,
                        next_batch,
                        weights,
                        step_choices)

            def update_priorities(self, idxes, priorities):
                priorities = priorities.reshape((self.n_envs, int(self.batch_size / self.n_envs)))
                for i in range(self.n_envs):
                    for idx, priority in zip(idxes[i], priorities[i]):
                        if priority == 0:
                            priority += 1e-8
                        assert 0 <= idx < self.size
                        self._it_sum[i][idx] = priority ** self._alpha
                        self._it_min[i][idx] = priority ** self._alpha

                        self._max_priority[i] = max(self._max_priority[i], priority)


        class DummyOffPolicyBuffer_Atari(DummyOffPolicyBuffer):
            """
            Replay buffer for off-policy DRL algorithms and Atari tasks.
                observation_space: the observation space of the environment.
                action_space: the action space of the environment.
                auxiliary_shape: data shape of auxiliary information (if exists).
                n_envs: number of parallel environments.
                n_size: max length of steps to store for one environment.
                batch_size: batch size of transition data for a sample.
            """
            def __init__(self,
                         observation_space: Space,
                         action_space: Space,
                         auxiliary_shape: Optional[dict],
                         n_envs: int,
                         n_size: int,
                         batch_size: int):
                super(DummyOffPolicyBuffer_Atari, self).__init__(observation_space, action_space, auxiliary_shape,
                                                                 n_envs, n_size, batch_size)
                self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size, np.uint8)
                self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size, np.uint8)

            def clear(self):
                self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size, np.uint8)
                self.next_observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size, np.uint8)
                self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
                self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)
                self.rewards = create_memory((), self.n_envs, self.n_size)
                self.terminals = create_memory((), self.n_envs, self.n_size)


        class DummyOnPolicyBuffer_Atari(DummyOnPolicyBuffer):
            """
            Replay buffer for on-policy DRL algorithms and Atari tasks.
                observation_space: the observation space of the environment.
                action_space: the action space of the environment.
                auxiliary_shape: data shape of auxiliary information (if exists).
                n_envs: number of parallel environments.
                n_size: max length of steps to store for one environment.
                use_gae: if use GAE trick.
                use_advnorm: if use Advantage normalization trick.
                gamma: discount factor.
                gae_lam: gae lambda.
            """
            def __init__(self,
                         observation_space: Space,
                         action_space: Space,
                         auxiliary_shape: Optional[dict],
                         n_envs: int,
                         n_size: int,
                         use_gae: bool = True,
                         use_advnorm: bool = True,
                         gamma: float = 0.99,
                         gae_lam: float = 0.95):
                super(DummyOnPolicyBuffer_Atari, self).__init__(observation_space, action_space, auxiliary_shape,
                                                                n_envs, n_size, use_gae, use_advnorm, gamma, gae_lam)
                self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size, np.uint8)

            def clear(self):
                self.ptr, self.size = 0, 0
                self.observations = create_memory(space2shape(self.observation_space), self.n_envs, self.n_size, np.uint8)
                self.actions = create_memory(space2shape(self.action_space), self.n_envs, self.n_size)
                self.auxiliary_infos = create_memory(self.auxiliary_shape, self.n_envs, self.n_size)
                self.rewards = create_memory((), self.n_envs, self.n_size)
                self.returns = create_memory((), self.n_envs, self.n_size)
                self.advantages = create_memory((), self.n_envs, self.n_size)

  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python



