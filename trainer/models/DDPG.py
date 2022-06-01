import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from trainer.replay_buffer import ReplayBuffer
from trainer.utils import encode_obs


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc_1 = nn.Linear(obs_dim, hidden_dim)
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act_2 = nn.ReLU()
        self.fc_3 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.act_3 = nn.ReLU()
        self.fc_4 = nn.Linear(hidden_dim, action_dim)
        self.act_4 = nn.Tanh()

    def forward(self, obs):
        x = self.fc_1(obs)
        x = self.act_1(x)
        x = self.fc_2(x)
        x = self.act_2(x)
        x = self.fc_3(x)
        x = self.act_3(x)
        x = self.fc_4(x)
        x = self.act_4(x)
        return x


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc_1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.act_2 = nn.ReLU()
        self.fc_3 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.act_3 = nn.ReLU()
        self.fc_4 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=-1)
        x = self.fc_1(x)
        x = self.act_1(x)
        x = self.fc_2(x)
        x = self.act_2(x)
        x = self.fc_3(x)
        x = self.act_3(x)
        x = self.fc_4(x)
        return x


class DDPG:
    def __init__(self, obs_dim, hidden_dim, action_dim, args):
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.args = args
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed

        self.actor = Actor(obs_dim, hidden_dim, action_dim)
        self.actor_target = Actor(obs_dim, hidden_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.a_lr)

        self.critic = Critic(obs_dim, hidden_dim, action_dim)
        self.critic_target = Critic(obs_dim, hidden_dim, action_dim)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.c_lr)

        self.hard_update()

        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)

        self.a_loss = None
        self.c_loss = None

    def hard_update(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def soft_update(self):
        for src_param, tgt_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            tgt_param.data.copy_(
                tgt_param.data * (1.0 - self.tau) + src_param.data * self.tau
            )
        for src_param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(
                tgt_param.data * (1.0 - self.tau) + src_param.data * self.tau
            )

    def choose_action(self, obs, action_space=None):
        obs = encode_obs(obs)
        if action_space is None:
            p = np.random.random()
            if p > self.eps:
                obs = torch.tensor([obs])
                action = self.actor(obs).detach().numpy()[0]
            else:
                action = np.random.uniform(-1, 1, self.action_dim)
            return action.argmax(), action
        else:
            # Just for discard and replace
            replace_num = 0
            for i in action_space:
                if i.action_type == "replace" or i.action_type == "discard":
                    replace_num += 1
            p = np.random.random()
            if p > self.eps:
                obs = torch.tensor([obs])
                action = self.actor(obs).detach().numpy()[0]
            else:
                action = np.random.uniform(-1, 1, self.action_dim)
            action_origin = action.copy()
            while True:
                try:
                    action_temp = action.argmax()
                except ValueError:
                    return 0, action_origin
                if action_temp // 9 == 0:
                    tile_rank = action_temp + 11
                elif action_temp // 9 == 1:
                    tile_rank = action_temp + 12
                elif action_temp // 9 == 2:
                    tile_rank = action_temp + 13
                elif action_temp // 9 == 3:
                    tile_rank = action_temp + 14
                for i in action_space[0: replace_num]:
                    if i.action_string == str(tile_rank):
                        return action_space.index(i), action_origin
                action = np.delete(action, action_temp)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batches()

        state_batch = torch.Tensor(state_batch).reshape(
            self.batch_size, 1, -1)
        action_batch = torch.Tensor(action_batch).reshape(
            self.batch_size, 1, -1)
        reward_batch = torch.Tensor(reward_batch).reshape(
            self.batch_size, 1, 1)
        next_state_batch = torch.Tensor(next_state_batch).reshape(
            self.batch_size, 1, -1)
        done_batch = torch.Tensor(done_batch).reshape(
            self.batch_size, 1, 1)

        with torch.no_grad():
            target_next_actions = self.actor_target(next_state_batch)
            target_next_q = self.critic_target(
                next_state_batch, target_next_actions)
            q_hat = reward_batch + self.gamma * \
                target_next_q * (1 - done_batch)

        main_q = self.critic(state_batch, action_batch)
        loss_critic = torch.nn.MSELoss()(q_hat, main_q)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        loss_actor = -self.critic(state_batch, self.actor(state_batch)).mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        self.c_loss = loss_critic.item()
        self.a_loss = loss_actor.item()

        self.soft_update()
        self.soft_update()

        return self.c_loss, self.a_loss

    def save(self, name):
        torch.save(self.actor.state_dict(), name + "_actor.pth")
        torch.save(self.critic.state_dict(), name + "_critic.pth")
