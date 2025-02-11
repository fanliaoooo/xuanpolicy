MFQ_Learner
=====================================

xxxxxx.

.. raw:: html

    <br><hr>

**PyTorch:**

.. py:class::
  xuance.torch.learners.multi_agent_rl.mfq_learner.MFQ_Learner(config, policy, optimizer, scheduler, device, model_dir, gamma, sync_frequency)

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
  xuance.torch.learners.multi_agent_rl.mfq_learner.MFQ_Learner.get_boltzmann_policy(q)

  xxxxxx.

  :param sample: xxxxxx.
  :type sample: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.torch.learners.multi_agent_rl.mfq_learner.MFQ_Learner.update(sample)

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
  xuance.mindspore.learners.multi_agent_rl.mfq_learner.MFQ_Learner(config, policy, optimizer, scheduler, model_dir, gamma, sync_frequency)

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
  xuance.mindspore.learners.multi_agent_rl.mfq_learner.MFQ_Learner.get_boltzmann_policy(q)

  xxxxxx.

  :param sample: xxxxxx.
  :type sample: xxxxxx
  :return: xxxxxx.
  :rtype: xxxxxx

.. py:function::
  xuance.mindspore.learners.multi_agent_rl.mfq_learner.MFQ_Learner.update(sample)

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
        MFQ: Mean Field Q-Learning
        Paper link:
        http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
        Implementation: Pytorch
        """
        from xuance.torch.learners import *


        class MFQ_Learner(LearnerMAS):
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
                self.temperature = config.temperature
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                self.softmax = torch.nn.Softmax(dim=-1)
                super(MFQ_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)

            def get_boltzmann_policy(self, q):
                return self.softmax(q / self.temperature)

            def update(self, sample):
                self.iterations += 1
                obs = torch.Tensor(sample['obs']).to(self.device)
                actions = torch.Tensor(sample['actions']).to(self.device)
                obs_next = torch.Tensor(sample['obs_next']).to(self.device)
                act_mean = torch.Tensor(sample['act_mean']).to(self.device)
                act_mean_next = torch.Tensor(sample['act_mean_next']).to(self.device)
                rewards = torch.Tensor(sample['rewards']).to(self.device)
                terminals = torch.Tensor(sample['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
                agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
                IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

                act_mean = act_mean.unsqueeze(1).repeat([1, self.n_agents, 1])
                act_mean_next = act_mean_next.unsqueeze(1).repeat([1, self.n_agents, 1])
                _, _, q_eval = self.policy(obs, act_mean, IDs)
                q_eval_a = q_eval.gather(-1, actions.long().reshape([self.args.batch_size, self.n_agents, 1]))
                q_next = self.policy.target_Q(obs_next, act_mean_next, IDs)
                shape = q_next.shape
                pi = self.get_boltzmann_policy(q_next)
                v_mf = torch.bmm(q_next.reshape(-1, 1, shape[-1]), pi.unsqueeze(-1).reshape(-1, shape[-1], 1))
                v_mf = v_mf.reshape(*(list(shape[0:-1]) + [1]))
                q_target = rewards + (1 - terminals) * self.args.gamma * v_mf

                # calculate the loss function
                td_error = (q_eval_a - q_target.detach()) * agent_mask
                loss = (td_error ** 2).sum() / agent_mask.sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()

                lr = self.optimizer.state_dict()['param_groups'][0]['lr']

                info = {
                    "learning_rate": lr,
                    "loss_Q": loss.item(),
                    "predictQ": q_eval_a.mean().item()
                }

                return info













  .. group-tab:: TensorFlow

    .. code-block:: python


  .. group-tab:: MindSpore

    .. code-block:: python

        """
        MFQ: Mean Field Q-Learning
        Paper link:
        http://proceedings.mlr.press/v80/yang18d/yang18d.pdf
        Implementation: MindSpore
        """
        from xuance.mindspore.learners import *


        class MFQ_Learner(LearnerMAS):
            class PolicyNetWithLossCell(nn.Cell):
                def __init__(self, backbone, n_agents):
                    super(MFQ_Learner.PolicyNetWithLossCell, self).__init__()
                    self._backbone = backbone
                    self.n_agents = n_agents

                def construct(self, bs, o, a, a_mean, agt_mask, ids, tar_q):
                    _, _, q_eval = self._backbone(o, a_mean, ids)
                    q_eval_a = GatherD()(q_eval, -1, a.astype(ms.int32).view(bs, self.n_agents, 1))
                    td_error = (q_eval_a - tar_q) * agt_mask
                    loss = (td_error ** 2).sum() / agt_mask.sum()
                    return loss

            def __init__(self,
                         config: Namespace,
                         policy: nn.Cell,
                         optimizer: nn.Optimizer,
                         scheduler: Optional[nn.exponential_decay_lr] = None,
                         summary_writer: Optional[SummaryWriter] = None,
                         model_dir: str = "./",
                         gamma: float = 0.99,
                         sync_frequency: int = 100
                         ):
                self.gamma = gamma
                self.temperature = config.temperature
                self.sync_frequency = sync_frequency
                self.mse_loss = nn.MSELoss()
                self.softmax = nn.Softmax(axis=-1)
                super(MFQ_Learner, self).__init__(config, policy, optimizer, scheduler, model_dir)
                self.bmm = ops.BatchMatMul()
                self.loss_net = self.PolicyNetWithLossCell(policy, self.n_agents)
                self.poliy_train = nn.TrainOneStepCell(self.loss_net, optimizer)
                self.poliy_train.set_train()

            def get_boltzmann_policy(self, q):
                return self.softmax(q / self.temperature)

            def update(self, sample):
                self.iterations += 1
                obs = Tensor(sample['obs'])
                actions = Tensor(sample['actions'])
                obs_next = Tensor(sample['obs_next'])
                act_mean = Tensor(sample['act_mean'])
                act_mean_next = Tensor(sample['act_mean_next'])
                rewards = Tensor(sample['rewards'])
                terminals = Tensor(sample['terminals']).view(-1, self.n_agents, 1)
                agent_mask = Tensor(sample['agent_mask']).view(-1, self.n_agents, 1)
                batch_size = obs.shape[0]
                IDs = ops.broadcast_to(self.expand_dims(self.eye(self.n_agents, self.n_agents, ms.float32), 0),
                                       (batch_size, -1, -1))

                act_mean = ops.broadcast_to(self.expand_dims(act_mean, 1), (-1, self.n_agents, -1))
                act_mean_next = ops.broadcast_to(self.expand_dims(act_mean_next, 1), (-1, self.n_agents, -1))
                q_next = self.policy.target_Q(obs_next, act_mean_next, IDs)
                shape = q_next.shape
                pi = self.get_boltzmann_policy(q_next)
                v_mf = self.bmm(q_next.view(-1, 1, shape[-1]), self.expand_dims(pi, -1).view(-1, shape[-1], 1))
                v_mf = v_mf.view(tuple(list(shape[0:-1]) + [1]))
                q_target = rewards + (1 - terminals) * self.args.gamma * v_mf

                # calculate the loss function
                loss = self.poliy_train(batch_size, obs, actions, act_mean, agent_mask, IDs, q_target)
                if self.iterations % self.sync_frequency == 0:
                    self.policy.copy_target()

                lr = self.scheduler(self.iterations).asnumpy()

                info = {
                    "learning_rate": lr,
                    "loss_Q": loss.asnumpy()
                }

                return info


