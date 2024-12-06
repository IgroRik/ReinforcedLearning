from collections import deque
import random
import time
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imageio

device = torch.device("cpu")

# Modified policy for BipedalWalker
class Policy(nn.Module):
    def __init__(self, s_size=24, h_size=32, a_size=4):  # 4 actions for BipedalWalker
        super(Policy, self).__init__()
        self.a_size = a_size

        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self, state, epsilon=0.0):
        print(f"State before action: {state}, type: {type(state)}, shape: {getattr(state, 'shape', 'No shape attribute')}")
        
        # Check for nested structures
        if isinstance(state, tuple):
            state = state[0]  # Extract array from tuple if needed
        
        state = np.asarray(state)  # Convert to numpy array

        if np.random.rand() <= epsilon:
            return np.random.choice(self.a_size, size=4)  # Generate random actions
        
        q_values = self(torch.tensor(state, dtype=torch.float32, device=device)).to("cpu")
        return np.clip(q_values.detach().numpy(), -1, 1)  # Clip actions within the range [-1, 1]





class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, epsilon_decay, buffer_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.model = Policy(s_size=state_dim, a_size=action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(torch.tensor(next_state, dtype=torch.float32, device=device))).item()

            with torch.no_grad():
                target_f = self.model(torch.tensor(state, dtype=torch.float32, device=device)).to("cpu").numpy()
            
            # Convert action to indices and update the corresponding elements in target_f
                for i in range(len(action)):
                    target_f[i] = target  # Update each action element with a separate value

                target_f = torch.tensor(target_f, dtype=torch.float32, device=device)

            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, self.model(torch.tensor(state, dtype=torch.float32, device=device)))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay


def test(env, model, policy=None, render=False, num_episodes=5):
    print(f"Policy received in test: {policy}, type: {type(policy)}")
    if policy is None:
        raise ValueError("Policy is None! Ensure you pass a valid policy object to the test function.")

    total_reward = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state)  # Convert state to numpy array
        episode_reward = 0

        done, truncated = False, False
        while not (done or truncated):
            if render:
                env.render()

            try:
                action = policy.act(state)  # Get action from policy
            except Exception as e:
                print(f"Error in policy.act at episode {episode+1}: {e}")
                break

            next_state, reward, done, truncated, info = env.step(action)
            next_state = np.array(next_state)  # Convert next state to numpy array

            episode_reward += reward
            state = next_state

        total_reward += episode_reward
        print(f"Test | Episode: {episode+1} | Episode Reward: {episode_reward}")

    return total_reward / num_episodes



def train(env):
    state_dim = env.observation_space.shape[0]  # State space dimension
    action_dim = env.action_space.shape[0]      # Number of actions (4 for BipedalWalker)
    agent = DQNAgent(state_dim, action_dim, lr=0.0005, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=1000)

    scores = []
    scores_deque = deque(maxlen=100)

    best_reward = -np.inf

    for i_episode in range(200):
        state, _ = env.reset()  # Get state and additional info
        # Check state
        if isinstance(state, np.ndarray):  # If state is a numpy array
            state = state.flatten()  # Flatten it to a 1D array
        else:
            print(f"State is not a numpy array: {state}")

        print(f"State shape after reset: {state.shape}")  # Output the shape of the state
        score = 0
        for t in range(1000):
            action = agent.model.act(state, epsilon=agent.epsilon)  # Return a vector of 4 values now
            action = np.clip(action, -1, 1)  # Clip actions within the range [-1, 1]
            next_state, reward, done, truncated, info = env.step(action)  # Now action is a vector of 4 values

            # Check state
            if isinstance(next_state, np.ndarray):
                next_state = next_state.flatten()  # Flatten it to a 1D array

            print(f"Next state shape: {next_state.shape}")  # Output the shape of the next state

            agent.remember(state, action, reward, next_state, done)

            state = next_state
            score += reward

            agent.replay(16)

            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        print(f'Episode {i_episode} Score {score} Average Score {np.mean(scores_deque)}')

        if i_episode % 30 == 29:
            reward = test(env, agent.model, policy=agent, render=False, num_episodes=5)

            if reward > best_reward:
                best_reward = reward
                torch.save(agent.model.state_dict(), 'checkpoint_best.pth')




if __name__ == '__main__':
    env = gym.make('BipedalWalker-v3', render_mode="rgb_array")
    env.reset(seed=0)

    checkpoint = torch.load('checkpoint_best.pth')
    print("Checkpoint loaded:", checkpoint)

    train(env)  # Training the agent

    # Load the model with the best parameters
    agent_policy = Policy(s_size=24, a_size=4)

    try:
        checkpoint = torch.load('checkpoint_best.pth')
        print(f"Checkpoint keys: {checkpoint.keys()}")  # Check that the keys match the model parameters
        agent_policy.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        agent_policy = None  # Explicitly set policy in case of error

    if agent_policy is not None:
        print("Starting test...")
        test(env=env, model=agent_policy, policy=agent_policy, render=False, num_episodes=25)
        test(env=env, model=agent_policy, policy=agent_policy, render=True, num_episodes=5)
    else:
        print("Policy is None. Skipping test.")

    # test(env, policy, render=False, num_episodes=25)  # Test without rendering
    # test(env, policy, render=True)  # Test with rendering
