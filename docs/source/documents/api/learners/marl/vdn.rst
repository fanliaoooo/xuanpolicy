VDN_Learner
=====================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.multi_agent_rl.vdn_learner.VDN_Learner(config, policy, optimizer, scheduler, device, model_dir, gamma, sync_frequency)

  :param config: xxxxxx.
  :type config: xxxxxx
  :param policy: xxxxxx.
  :type policy: xxxxxx
  :param optimizer: xxxxxx.
  :type optimizer: xxxxxx
  :param scheduler: xxxxxx.
  :type scheduler: xxxxxx
  :param device: xxxxxx.
  :type device: xxxxxx
  :param model_dir: xxxxxx.
  :type model_dir: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx
  :param sync_frequency: xxxxxx.
  :type sync_frequency: xxxxxx

.. py:function::
  xuance.torch.learners.multi_agent_rl.vdn_learner.VDN_Learner.update(sample)

  xxxxxx.

  :param sample: xxxxxx.
  :type sample: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.learners.multi_agent_rl.vdn_learner.VDN_Learner.update_recurrent(sample)

  xxxxxx.

  :param sample: xxxxxx.
  :type sample: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

**TensorFlow:**

.. raw:: html

    <br><hr>

**MindSpore:**

.. py:class::
  xuance.mindspore.learners.multi_agent_rl.vdn_learner.VDN_Learner(config, policy, optimizer, scheduler, model_dir, gamma, sync_frequency)

  :param config: xxxxxx.
  :type config: xxxxxx
  :param policy: xxxxxx.
  :type policy: xxxxxx
  :param optimizer: xxxxxx.
  :type optimizer: xxxxxx
  :param scheduler: xxxxxx.
  :type scheduler: xxxxxx
  :param model_dir: xxxxxx.
  :type model_dir: xxxxxx
  :param gamma: xxxxxx.
  :type gamma: xxxxxx
  :param sync_frequency: xxxxxx.
  :type sync_frequency: xxxxxx

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.vdn_learner.VDN_Learner.update(sample)

  xxxxxx.

  :param sample: xxxxxx.
  :type sample: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. raw:: html

    <br><hr>

Source Code
-----------------

.. tabs::

  .. group-tab:: PyTorch

    .. code-block:: python

        """
        Value Decomposition Networks (VDN)
        Paper link:
        https://arxiv.org/pdf/1706.05296.pdf
        Implementation: Pytorch
        """
        from xuance.torch.learners import *


        class VDN_Learner(LearnerMAS):
            def __init__(self,
                         config: Namespace,
                         policy: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                         device: Optional[Union[int, str, torch.device]] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                self.use_recurrent = config.use_recurrent
                self.mse_loss = nn.MSELoss()
                super(VDN_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)

            def update(self, sample):
                self.iterations += 1
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                obs_next = torch.Tensor(sample['obs_next']).to(self.device)
                rewards = torch.Tensor(sample['rewards']).mean(dim=1).to(self.device)
                terminals = torch.Tensor(sample['terminals']).all(dim=1, keepdims=True).float().to(self.device)
                agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

                _, _, q_eval = self.policy(obs, IDs)
                q_eval_a = q_eval.gather(-1, actions.long().reshape([self.args.batch_size, self.n_agents, 1]))
                q_tot_eval = self.policy.Q_tot(q_eval_a * agent_mask)
                _, q_next = self.policy.target_Q(obs_next, IDs)
                if self.args.double_q:
                    _, action_next_greedy, _ = self.policy(obs_next, IDs)
                    q_next_a = q_next.gather(-1, action_next_greedy.unsqueeze(-1).long().detach())
                else:
                    q_next_a = q_next.max(dim=-1, keepdim=True).values
                q_tot_next = self.policy.target_Q_tot(q_next_a * agent_mask)
                q_tot_target = rewards + (1 - terminals) * self.args.gamma * q_tot_next

                # calculate the loss function
                loss = self.mse_loss(q_tot_eval, q_tot_target.detach())
                self.optimizer.zero_grad()
                loss.backward()
                if self.args.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "learning_rate": lr,
                    "loss_Q": loss.item(),
                    "predictQ": q_tot_eval.mean().item()
                }

                return info

            def update_recurrent(self, sample):
                """
                Update the parameters of the model with recurrent neural networks.
                """
                self.iterations += 1
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                state = torch.Tensor(sample['state']).to(self.device)
                rewards = torch.Tensor(sample['rewards']).mean(dim=1, keepdims=False).to(self.device)
                terminals = torch.Tensor(sample['terminals']).float().to(self.device)
                avail_actions = torch.Tensor(sample['avail_actions']).float().to(self.device)
                filled = torch.Tensor(sample['filled']).float().to(self.device)
                batch_size = actions.shape[0]
                episode_length = actions.shape[2]
                IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, episode_length + 1, -1).to(
                    self.device)

                # Current Q
                rnn_hidden = self.policy.representation.init_hidden(batch_size * self.n_agents)
                _, actions_greedy, q_eval = self.policy(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                                        IDs.reshape(-1, episode_length + 1, self.n_agents),
                                                        *rnn_hidden,
                                                        avail_actions=avail_actions.reshape(-1, episode_length + 1, self.dim_act))
                q_eval = q_eval[:, :-1].reshape(batch_size, self.n_agents, episode_length, self.dim_act)
                actions_greedy = actions_greedy.reshape(batch_size, self.n_agents, episode_length + 1, 1)
                q_eval_a = q_eval.gather(-1, actions.long().reshape([self.args.batch_size, self.n_agents, episode_length, 1]))
                q_tot_eval = self.policy.Q_tot(q_eval_a) * filled

                # Target Q
                target_rnn_hidden = self.policy.target_representation.init_hidden(batch_size * self.n_agents)
                _, q_next = self.policy.target_Q(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                                 IDs.reshape(-1, episode_length + 1, self.n_agents),
                                                 *target_rnn_hidden)
                q_next = q_next[:, 1:].reshape(batch_size, self.n_agents, episode_length, self.dim_act)
                q_next[avail_actions[:, :, 1:] == 0] = -9999999

                # use double-q trick
                if self.args.double_q:
                    action_next_greedy = actions_greedy[:, :, 1:]
                    q_next_a = q_next.gather(-1, action_next_greedy.long().detach())
                else:
                    q_next_a = q_next.max(dim=-1, keepdim=True).values

                q_tot_next = self.policy.target_Q_tot(q_next_a) * filled
                rewards *= filled
                q_tot_target = rewards + (1 - terminals) * self.args.gamma * q_tot_next

                # calculate the loss function
                td_errors = q_tot_eval - q_tot_target.detach()
                loss = (td_errors ** 2).sum() / filled.sum()
                self.optimizer.zero_grad()
                loss.backward()
                if self.args.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip_norm)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()
                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "learning_rate": lr,
                    "loss_Q": loss.item(),
                    "predictQ": q_tot_eval.mean().item()
                }

                return info

  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python

        """
        Value Decomposition Networks (VDN)
        Paper link:
        https://arxiv.org/pdf/1706.05296.pdf
        Implementation: MindSpore
        """
        from xuance.mindspore.learners import *


        class VDN_Learner(LearnerMAS):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone):
                    super(VDN_Learner.PolicyNetWithLossCell, self).__init__(auto_prefix=False)
                    self._backbone = backbone

                def construct(self, o, ids, a, label, agt_mask):
                    _, _, q_eval = self._backbone(o, ids)
                    q_eval_a = GatherD()(q_eval, -1, a)
                    q_tot_eval = self._backbone.Q_tot(q_eval_a * agt_mask)
                    td_error = q_tot_eval - label
                    loss = (td_error ** 2).sum() / agt_mask.sum()
                    return loss

            def __init__(self,
                         config: Namespace,
                         policy: nn.Cell,
                         optimizer: nn.Optimizer,
                         scheduler: Optional[nn.exponential_decay_lr] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.gamma = gamma
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                super(VDN_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
                # build train net
                self._mean = ops.ReduceMean(keep_dims=False)
                self.loss_net = self.PolicyNetWithLossCell(policy)
                self.policy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
                self.policy_train.set_train()

            def update(self, sample):
                self.iterations += 1
                obs = Tensor(sample['obs'])
                actions = Tensor(sample['actions']).view(-1, self.n_agents, 1).astype(ms.int32)
                obs_next = Tensor(sample['obs_next'])
                rewards = self._mean(Tensor(sample['rewards']), 1)
                terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1).all(axis=1, keep_dims=True).astype(ms.float32)
                agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
                batch_size = obs.shape[0]
                IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                       (batch_size, -1, -1))
                # calculate the target total values
                _, q_next = self.policy.target_Q(obs_next, IDs)
                if self.args.double_q:
                    _, action_next_greedy, _ = self.policy(obs_next, IDs)
                    action_next_greedy = self.expand_dims(action_next_greedy, -1).astype(ms.int32)
                    q_next_a = GatherD()(q_next, -1, action_next_greedy)
                else:
                    q_next_a = q_next.max(axis=-1, keepdims=True).values
                q_tot_next = self.policy.target_Q_tot(q_next_a * agent_mask)
                q_tot_target = rewards + (1-terminals) * self.args.gamma * q_tot_next

                # calculate the loss and train
                loss = self.policy_train(obs, IDs, actions, q_tot_target, agent_mask)
                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()

                lr = self.scheduler(self.iterations).asnumpy()

                info = {
                    "learning_rate": lr,
                    "loss_Q": loss.asnumpy()
                }

                return info
