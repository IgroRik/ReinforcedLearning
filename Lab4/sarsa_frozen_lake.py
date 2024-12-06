import gym
import random
from collections import defaultdict
import numpy as np
import imageio

def show_agent_gif(env, policy, filename="Lab4/frozen_lake_policy_sarsa.gif"):
    frames = []
    state, _ = env.reset()  # Get the initial state
    done = False

    while not done:
        action = policy[state]  # Get the action based on the optimal policy
        next_state, reward, done, _, _ = env.step(action)  # Perform the action

        # Add a frame to the GIF at each step
        frame = env.render()  # Render the game field as an image
        frames.append(frame)

        state = next_state  # Update the state

    # Create a GIF that shows the agent's path
    imageio.mimsave(filename, frames, duration=0.5)
    print(f"GIF saved to {filename}")


def epsilon_greedy(Q, state, epsilon, env):
    """Returns an action based on epsilon-greedy policy"""
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Random action
    else:
        return max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])


def generate_policy(Q, env):
    """Generates an optimal policy based on Q-function"""
    policy = defaultdict(int)
    for state in range(env.observation_space.n):
        policy[state] = max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])
    return policy

def sarsa(env, num_episodes, num_timesteps, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay):
    """SARSA algorithm for training the agent with epsilon decay"""
    Q = defaultdict(float)  # Initialize the Q-function
    epsilon = epsilon_start

    for i in range(num_episodes):
        state, _ = env.reset()  # Extract the initial state
        action = epsilon_greedy(Q, state, epsilon, env)  # Choose the initial action

        for t in range(num_timesteps):
            next_state, reward, done, _, _ = env.step(action)  # Perform the action
            next_action = epsilon_greedy(Q, next_state, epsilon, env)  # Choose the next action

            # Update the Q-value for the current state-action pair
            Q[(state, action)] += alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])

            state = next_state
            action = next_action

            if done:
                break  # End the episode if the agent reaches the goal or falls into the hole

        # Decay epsilon after each episode
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    return Q

def test(env, policy, render=True):
    """Tests the learned policy"""
    state, _ = env.reset()  # Extract the initial state
    if render:
        env.render()

    total_reward = 0
    for _ in range(1000):  # Maximum number of steps
        action = policy[state]  # Choose action based on the policy
        next_state, reward, done, _, _ = env.step(action)

        if render:
            env.render()

        total_reward += reward
        state = next_state
        if done:
            break  # End the episode if done

    return total_reward

def show_agent(env, policy):
    """Shows the agent's behavior based on the learned policy"""
    state, _ = env.reset()  # Corrected: only the initial state
    env.render()

    for t in range(1000):  # Maximum number of steps
        action = policy[state]  # Choose action based on the policy
        next_state, reward, done, _, info = env.step(action)  # Corrected: 5 return values

        env.render()
        
        state = next_state  # Update the state

        if done:
            break  # End the episode if done

    env.close()

def generate_random_policy(env):
    """Generates a random policy"""
    policy = defaultdict(int)
    for state in range(env.observation_space.n):
        policy[state] = env.action_space.sample()
    return policy

def show_Q(env, Q):
    """Displays Q-values for each state and action"""
    print("************************************")
    for action in range(env.action_space.n):
        table = np.array([Q[(state, action)] for state in range(env.observation_space.n)])
        print(table.reshape(4,4))

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', is_slippery=True, desc=None, map_name="4x4", render_mode="rgb_array")  # с is_slippery=True
    env.reset(seed=42)

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.9

    num_episodes = 5000
    num_timesteps = 10000

    epsilon_start = 0.9
    epsilon_end = 0.1
    epsilon_decay = 0.999

    Q = sarsa(env, num_episodes, num_timesteps, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay)


    policy = generate_policy(Q, env)

    print("Optimal policy:")
    print(policy)


    show_agent(env, policy)


    sum_reward = 0
    for _ in range(5000):
        total_reward = test(env, policy, render=False)
        sum_reward += total_reward

    print(f"Average reward for 5000 episodes: {sum_reward / 5000}")
    show_agent_gif(env, policy)

    env.close()
