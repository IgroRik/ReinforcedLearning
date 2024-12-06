import gym
import numpy as np
import random
from collections import defaultdict
import imageio

# Epsilon-greedy policy function
def epsilon_greedy_policy(state, Q, epsilon=0.1):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Random action
    else:
        return max(range(env.action_space.n), key=lambda x: Q[(state, x)])  # Best policy

# Discretize the state
def discretize_state(state, bins):
    state_idx = []
    for i in range(len(state)):
        state_idx.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_idx)

# Generate an episode
def generate_episode(env, Q, bins, max_timesteps=100):
    episode = []
    state = env.reset()[0]  # Get the initial state

    for t in range(max_timesteps):
        state_idx = discretize_state(state, bins)  # Discretize the state
        action = epsilon_greedy_policy(state_idx, Q)  # Select action using epsilon-greedy policy

        next_state, reward, done, _, _ = env.step(action)  # Perform action
        episode.append((state_idx, action, reward))  # Save the step of the episode

        if done:
            break

        state = next_state  # Update state

    return episode

# Generate policy from Q-values
def generate_policy(Q, bins):
    policy = defaultdict(int)
    states = set(state for (state, action) in Q.keys())
    for state in states:
        policy[state] = max(range(env.action_space.n), key=lambda x: Q[(state, x)])
    return policy

# Test the policy
def test_policy(env, policy, bins, num_episodes=100, max_timesteps=1000):
    total_reward = 0
    for _ in range(num_episodes):
        state = env.reset()[0]  # Get the initial state
        state_idx = discretize_state(state, bins)
        episode_reward = 0

        for _ in range(max_timesteps):
            action = policy[state_idx]  # Select action from the policy
            next_state, reward, done, _, _ = env.step(action)  # Perform action
            episode_reward += reward

            if done:
                break

            state_idx = discretize_state(next_state, bins)  # Update state

        total_reward += episode_reward

    return total_reward / num_episodes  # Return average reward

# Function to save the training process as a GIF
def save_training_gif(env, policy, bins, filename="Lab3/trainingMC.gif", num_episodes=10, max_timesteps=1000):
    """
    Saves frames into a GIF for the most successful episode.

    Parameters:
    - env: Gym environment
    - policy: optimal policy
    - bins: state discretization
    - filename: file name to save the GIF
    - num_episodes: number of episodes for trial runs
    - max_timesteps: maximum number of steps per episode
    """
    best_episode_frames = None
    best_total_reward = float('-inf')

    # Run several episodes and select the best one
    for _ in range(num_episodes):
        frames = []
        state = env.reset()[0]  # Get the initial state
        done = False
        total_reward = 0
        timestep = 0

        while not done and timestep < max_timesteps:
            state_idx = discretize_state(state, bins)  # Discretize the state
            action = policy[state_idx]  # Get action from the policy

            # Perform the action
            next_state, reward, done, _, _ = env.step(action)

            # Capture frame for the GIF
            frame = env.render()  # Get the image in RGB format
            frames.append(frame)

            state = next_state  # Update state
            total_reward += reward
            timestep += 1

        # If the current episode is better, save its frames
        if total_reward > best_total_reward:
            best_total_reward = total_reward
            best_episode_frames = frames

    # Save the frames of the best episode as a GIF
    if best_episode_frames:
        imageio.mimsave(filename, best_episode_frames, duration=0.5)
        print(f"GIF saved to {filename}")
    else:
        print("No successful episode found for GIF.")

# Main part of the code
if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode="rgb_array")

    # Discretize the state
    position_bins = np.linspace(-1.2, 0.6, 24)
    velocity_bins = np.linspace(-0.07, 0.07, 24)
    bins = [position_bins, velocity_bins]

    # Initialize Q-table
    Q = defaultdict(float)
    total_return = defaultdict(float)
    N = defaultdict(int)

    num_iterations = 50000
    for _ in range(num_iterations):
        episode = generate_episode(env, Q, bins)

        # Extract all state-action pairs from the episode
        all_state_action_pairs = [(s, a) for (s, a, r) in episode]
        rewards = [r for (s, a, r) in episode]

        # For each step in the episode
        for t, (state, action, reward) in enumerate(episode):
            if not (state, action) in all_state_action_pairs[0:t]:
                # Compute the reward for state-action pair
                R = sum(rewards[t:])
                total_return[(state, action)] += R  # Update total return
                N[(state, action)] += 1  # Increment visit count

                # Update Q-value
                Q[(state, action)] = total_return[(state, action)] / N[(state, action)]

    # Generate policy from Q-values
    policy = generate_policy(Q, bins)

    # Test the generated policy
    avg_reward = test_policy(env, policy, bins)
    print(f"Average reward: {avg_reward}")
    save_training_gif(env, policy, bins)

    env.close()
