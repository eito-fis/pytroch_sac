import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from src.sac.sac_model import SACQNet, SACActor
from src.sac.sac_policy import SACPolicy
from src.general.policies.policy import RandomPolicy
from src.general.replay_buffers.replay_buffer import ReplayBuffer

class SACAgent():
    """
    SAC Agent class. Builds and trains a model
    """
    def __init__(self,
                 train_steps=None,
                 random_steps=None,
                 train_freq=1,
                 target_update_freq=1,
                 actor_lr=0.0042,
                 q_lr=0.0042,
                 entropy_lr=0.0042,
                 gamma=0.99,
                 alpha=1,
                 tau=0.005,
                 buffer_size=50000,
                 batch_size=256,
                 gradient_steps=1,
                 env=None,
                 actor_fc=None,
                 critic_fc=None,
                 conv_size=None,
                 device="cpu",
                 norm_reward=False,
                 logging_period=25,
                 checkpoint_period=50,
                 output_dir="/tmp/sac",
                 restore_dir=None,
                 wandb=None):

        # Build environment
        assert env
        self.env = env
        self.action_space = self.env.action_space

        # Build networks
        self.actor = SACActor(state_size=self.env.obs_space,
                              action_space=self.env.action_space,
                              fc=actor_fc,
                              conv_size=conv_size).to(device)
        self.q1 = SACQNet(state_size=self.env.obs_space,
                          action_space=self.env.action_space,
                          fc=critic_fc,
                          conv_size=conv_size).to(device)
        self.q2 = SACQNet(state_size=self.env.obs_space,
                          action_space=self.env.action_space,
                          fc=critic_fc,
                          conv_size=conv_size).to(device)
        self.q1_t = SACQNet(state_size=self.env.obs_space,
                            action_space=self.env.action_space,
                            fc=critic_fc,
                            conv_size=conv_size).to(device)
        self.q2_t = SACQNet(state_size=self.env.obs_space,
                            action_space=self.env.action_space,
                            fc=critic_fc,
                            conv_size=conv_size).to(device)
        self.hard_update(self.q1_t, self.q1)
        self.hard_update(self.q2_t, self.q2)

        # Build entropy parameters
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32).to(device)
        self.log_alpha.requires_grad = True
        self.alpha = torch.exp(self.log_alpha)
        self.target_entropy = -np.prod(self.env.action_space.shape)

        # Build policy, replay buffer and optimizers
        self.policy = SACPolicy(action_space=self.env.action_space,
                                model=self.actor,
                                device=device)
        self.random_policy = RandomPolicy(action_space=self.env.action_space,
                                          batch_size=1)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=q_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=q_lr)
        self.entropy_opt = torch.optim.Adam([self.log_alpha], lr=entropy_lr)

        # Setup training parameters
        self.gamma = gamma
        self.tau = tau
        self.train_steps = train_steps
        self.random_steps = random_steps
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps
        self.norm_reward = norm_reward

        self.device = device

        # Setup logging parameters
        self.reward_queue = deque(maxlen=100)
        self.logging_period = logging_period
        self.checkpoint_period = checkpoint_period
        self.episodes = 0
        self.wandb = wandb

        # Build logging directories
        self.log_dir = os.path.join(output_dir, "logs/")
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints/")
        os.makedirs(os.path.dirname(self.checkpoint_dir), exist_ok=True)

    def train(self):
        """
        Trains the model
        """
        obs = self.env.reset()
        for i in range(self.train_steps):
            if i < self.random_steps:
                action = self.random_policy()[0]
            else:
                _obs = torch.tensor(obs[None, :], dtype=torch.float32).to(self.device)
                action = self.policy.step(_obs)[0]
            assert action.shape == self.action_space.shape

            # Take step on env with action
            new_obs, rewards, done, self.infos = self.env.step(action)
            # self.env.render()
            # Store SARS(D) in replay buffer
            self.replay_buffer.add(obs, action, rewards, new_obs,
                                   float(done))
            obs = new_obs

            if done:
                self.reward_queue.extend([self.env.ep_reward])
                self.episodes += 1
                obs = self.env.reset()

            # Periodically learn
            if i % self.train_freq == 0:
                for g in range(self.gradient_steps):
                    # Don"t train if the buffer is not full enough or if we are
                    # still collecting random samples
                    if not self.replay_buffer.can_sample(self.batch_size) or \
                       i < self.random_steps:
                        break
                    self.update(i, g, done)

            # Periodically save models
            if i % self.checkpoint_period == 0 and i != 0:
                self.actor.save(f"actor_model_{i}", self.checkpoint_dir)
                self.q1.save(f"q1_model_{i}", self.checkpoint_dir)
                self.q2.save(f"q2_model_{i}", self.checkpoint_dir)
                self.q1_t.save(f"q1_t_model_{i}", self.checkpoint_dir)
                self.q2_t.save(f"q2_t_model_{i}", self.checkpoint_dir)

    def update(self, i, g, done, reward_scale=10):
        """
        Samples from the replay buffer and updates the model
        """
        # Sample and unpack batch
        batch = self.replay_buffer.sample(self.batch_size)
        b_obs, b_actions, b_rewards, b_n_obs, b_dones = batch
        b_rewards = b_rewards[:, None]
        b_dones = b_dones[:, None]
        b_obs, b_actions, b_rewards, b_n_obs, b_dones = \
                map(lambda x: torch.tensor(x, dtype=torch.float32).to(self.device),
                    (b_obs, b_actions, b_rewards, b_n_obs, b_dones))

        if self.norm_reward:
            reward = reward_scale * (reward - np.mean(reward, axis=0)) / \
                        (np.std(reward, axis=0) + 1e-6)

        # Update qs
        with torch.no_grad():
            b_n_actions, n_log_probs = self.policy.eval(b_n_obs)
            q1_ts = self.q1_t(b_n_obs, b_n_actions)
            q2_ts = self.q2_t(b_n_obs, b_n_actions)
            min_q_ts = torch.min(q1_ts, q2_ts) - self.alpha * n_log_probs
            target_q = b_rewards + (1 - b_dones) * self.gamma * min_q_ts

        q1s = self.q1(b_obs, b_actions)
        q1_loss = F.mse_loss(q1s, target_q)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        q2s = self.q2(b_obs, b_actions)
        q2_loss = F.mse_loss(q2s, target_q)
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # Update actor
        new_actions, log_probs = self.policy.eval(b_obs)
        n_q1s = self.q1(b_obs, new_actions)
        n_q2s = self.q2(b_obs, new_actions)
        min_n_qs = torch.min(n_q1s, n_q2s)
        actor_loss = torch.mean((self.alpha * log_probs) - min_n_qs)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update entropy constant
        entropy_loss = -torch.mean(self.log_alpha * \
                                   (log_probs + self.target_entropy).detach())
        self.entropy_opt.zero_grad()
        entropy_loss.backward()
        self.entropy_opt.step()

        self.alpha = torch.exp(self.log_alpha).detach()

        # Soft updates for the target network
        if (i + g) % self.target_update_freq == 0:
            self.soft_update(self.q1_t, self.q1)
            self.soft_update(self.q2_t, self.q2)

        if done:
            avg_prob = torch.mean(torch.exp(log_probs))
            self.log(actor_loss, avg_prob, q1_loss, q2_loss, entropy_loss,
                     i, g)

    def soft_update(self, q_t, q):
        """
        Soft update from q to target_q network based on self.tau
        """
        for t_w, w in zip(q_t.parameters(), q.parameters()):
            updated = (1 - self.tau) * t_w.data + self.tau * w.data
            t_w.data.copy_(updated)

    def hard_update(self, q_t, q):
        """
        Soft update from q to target_q network based on self.tau
        """
        for t_w, w in zip(q_t.parameters(), q.parameters()):
            t_w.data.copy_(w)

    def log(self, actor_loss, avg_prob, q1_loss, q2_loss, entropy_loss, i, g):
        # Periodically log
        if len(self.reward_queue) == 0:
            avg_reward = 0
        else:
            avg_reward = sum(self.reward_queue) / len(self.reward_queue)

        ep_reward = self.reward_queue[-1]
        print(f"Step {i} - Gradient Step {g}")
        print(f"| Episodes: {self.episodes} |")
        print(f"| Average Reward: {avg_reward} | Ep Reward: {ep_reward} |")
        print(f"| Actor Loss: {actor_loss} | Avg Probs: {avg_prob} |")
        print(f"| Q1 Loss: {q1_loss} | Q2 Loss: {q2_loss} |")
        print(f"| Entropy Loss: {entropy_loss} |")
        print(f"| Alpha: {self.alpha} |")
        print()

        if self.wandb != None:
            self.wandb.log({"Step": i,
                            "Average Reward": avg_reward,
                            "Episode Reward": ep_reward,
                            "Actor Loss": actor_loss.cpu().detach().numpy(),
                            "Average Prob": avg_prob.cpu().detach().numpy(),
                            "Entropy Loss": entropy_loss.cpu().detach().numpy(),
                            "Q1 Loss": q1_loss.detach().cpu().numpy(),
                            "Q2 Loss": q2_loss.detach().cpu().numpy(),
                            "Alpha": self.alpha.detach().cpu().numpy()})


if __name__ == "__main__":
    from src.general.envs.gym_env import GymEnv
    env = GymEnv("LunarLanderContinuous-v2")
    SACAgent(env=env,
             train_steps=1000000,
             random_steps=1000,
             train_freq=1,
             target_update_freq=1,
             actor_lr=0.0001,
             q_lr=0.0001,
             entropy_lr=0.001,
             gamma=0.99,
             alpha=1,
             tau=0.005,
             buffer_size=500000,
             batch_size=256,
             gradient_steps=1,
             actor_fc=(512, 256),
             critic_fc=(512, 256),
             conv_size=None,
             logging_period=25,
             checkpoint_period=5000)
