import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from collections import deque
import random

# 定义神经网络模型
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.actor(x), dim=-1)
        state_values = self.critic(x)
        return action_probs, state_values

# 定义 PPO 算法
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10, batch_size=64):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def update(self, states, actions, old_probs, rewards, dones):
        states = torch.FloatTensor(np.array(states))  # 转换为二维张量
        actions = torch.LongTensor(np.array(actions))  # 转换为一维张量
        old_probs = torch.FloatTensor(np.array(old_probs))  # 转换为一维张量
        rewards = torch.FloatTensor(np.array(rewards))  # 转换为一维张量
        dones = torch.FloatTensor(np.array(dones))  # 转换为一维张量

        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                batch_states = states[i:i+self.batch_size]
                batch_actions = actions[i:i+self.batch_size]
                batch_old_probs = old_probs[i:i+self.batch_size]
                batch_rewards = rewards[i:i+self.batch_size]
                batch_dones = dones[i:i+self.batch_size]

                action_probs, state_values = self.policy(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                entropy = dist.entropy().mean()

                new_probs = dist.log_prob(batch_actions).exp()
                ratio = new_probs / batch_old_probs

                advantages = batch_rewards - state_values.detach()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

                critic_loss = self.mse_loss(state_values, batch_rewards)

                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # 增加批次维度
        action_probs, _ = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), action_probs[0][action].item()  # 注意索引

# 训练函数
def train(env, ppo, episodes=1000, max_steps=500):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        if not isinstance(state, np.ndarray):  # 确保 state 是 NumPy 数组
            state = np.array(state)
        episode_reward = 0
        states, actions, old_probs, rewards_, dones = [], [], [], [], []

        for step in range(max_steps):
            action, old_prob = ppo.get_action(state)
            next_state, reward, done, _ = env.step(action)

            if not isinstance(next_state, np.ndarray):  # 确保 next_state 是 NumPy 数组
                next_state = np.array(next_state)

            states.append(state)
            actions.append(action)
            old_probs.append(old_prob)
            rewards_.append(reward)
            dones.append(done)

            state = next_state
            episode_reward += reward

            if done:
                break

        # 计算折扣奖励
        discounted_rewards = []
        cumulative_reward = 0
        for reward, done in zip(reversed(rewards_), reversed(dones)):
            cumulative_reward = reward + ppo.gamma * cumulative_reward * (1 - done)
            discounted_rewards.insert(0, cumulative_reward)

        # 更新 PPO
        ppo.update(states, actions, old_probs, discounted_rewards, dones)

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

    return rewards

# 主函数
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo = PPO(state_dim, action_dim)
    rewards = train(env, ppo)

    env.close()