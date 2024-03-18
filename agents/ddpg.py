import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from utils_folder import utils

class DDPGActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.trunk = utils.mlp(obs_dim, hidden_dim, action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        mu = self.trunk(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = utils.TruncatedNormal(mu, std)
        return dist
    
class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

class DDPG_Agent:
    """DDPG algorithm."""
    def __init__(self, obs_dim, net_action_dim, ctrl_dim, ctrl_horizon_dim, action_range,
                 device, hidden_dim, hidden_depth, discount, actor_lr, actor_betas, actor_update_frequency, 
                 critic_lr, critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, use_tb, num_expl_steps, stddev_schedule, stddev_clip):

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau

        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb

        self.ctrl_dim = ctrl_dim  
        self.ctrl_horizon_dim = ctrl_horizon_dim

        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size

        self.critic = DoubleQCritic(obs_dim[0], net_action_dim[0], hidden_dim, hidden_depth).to(self.device)
        self.critic_target = DoubleQCritic(obs_dim[0], net_action_dim[0], hidden_dim, hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DDPGActor(obs_dim[0], net_action_dim[0], hidden_dim, hidden_depth).to(self.device)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.train()
        self.critic_target.train()
        
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(self.action_range[0], self.action_range[1])
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done *self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.use_tb:
            metrics["critic_loss"] = critic_loss.item()
            metrics["critic_q1"] = current_Q1.mean().item()
            metrics["critic_q2"] = current_Q2.mean().item()
            metrics["critic_target_q"] = target_Q.mean().item()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_buffer, step):
        metrics = dict()

        _, obs, action, reward, _, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        metrics.update(self.update_critic(obs, action, reward, next_obs, not_done_no_max, step))

        if step % self.actor_update_frequency == 0:
            metrics.update(self.update_actor(obs, step))

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        return metrics
