import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network for policy approximation
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Function to calculate discounted rewards
def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards

# Training function
def train_cartpole():
    env = gym.make("CartPole-v1")
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    hidden_dim = 128
    learning_rate = 0.01
    gamma = 0.99
    num_episodes = 1000

    policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_rewards = []
        log_probs = []
        
        while not done:
            state = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_net(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            next_state, reward, done, _, _ = env.step(action.item())
            
            log_probs.append(log_prob)
            episode_rewards.append(reward)
            state = next_state
        
        # Compute discounted rewards
        discounted_rewards = compute_discounted_rewards(episode_rewards, gamma)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Compute loss
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.stack(policy_loss).sum()

        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Logging
        total_reward = sum(episode_rewards)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

        if total_reward >= 200:
            print("Solved CartPole!")
            break

    env.close()

if __name__ == "__main__":
    train_cartpole()
