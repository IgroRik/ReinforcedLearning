import gym
import random
import numpy as np
from collections import defaultdict
import imageio

def epsilon_greedy(Q, state, epsilon, env):
    """Returns an action based on epsilon-greedy policy"""
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()  # random action
    else:
        # action with the maximum Q-value for the current state
        return max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])

def generate_policy(Q, env):
    """Generates an optimal policy based on the Q-function"""
    policy = defaultdict(int)
    for state in range(env.observation_space.n):
        policy[state] = max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])
    return policy

def q_learning(env, num_episodes, num_timesteps, alpha, gamma, epsilon):
    """Q-learning algorithm for training the agent"""
    Q = defaultdict(float)

    for i in range(num_episodes):
        state, _ = env.reset()  # extract the initial state

        for t in range(num_timesteps):
            action = epsilon_greedy(Q, state, epsilon, env)  # choose an action
            next_state, reward, done, _, _ = env.step(action)  # perform the action

            # Update the Q-value
            Q[(state, action)] += alpha * (reward + gamma * np.max([Q[(next_state, a)] for a in range(env.action_space.n)]) - Q[(state, action)])

            state = next_state  # update the state

            if done:
                break  # end the episode if the agent reached the goal or fell into the hole

    return Q

def test(env, optimal_policy, render=True):
    """Tests the learned policy"""
    state, _ = env.reset()  # extract the initial state
    if render:
        env.render()

    total_reward = 0
    for _ in range(1000):  # maximum number of steps
        action = optimal_policy[state]  # choose an action based on the policy
        next_state, reward, done, _, _ = env.step(action)

        if render:
            env.render()

        total_reward += reward
        state = next_state
        if done:
            break  # end if the episode is done

    return total_reward

def show_agent_gif(env, policy, filename="Lab4/frozen_lake_policy_Q.gif"):
    """Creates a GIF with a successful completion"""
    frames = []
    state, _ = env.reset()
    done = False
    success = False

    while not done:
        action = policy[state]
        next_state, reward, done, _, _ = env.step(action)

        if reward == 1:  # If the reward is 1, it's a successful completion
            success = True

        # Capture a frame for the GIF
        frame = env.render()  # Render the game field
        frames.append(frame)

        state = next_state  # update the state

        if done:
            break

    if success:  # Only if the episode was successful
        # Save the frames as a GIF
        imageio.mimsave(filename, frames, duration=0.5)
        print(f"GIF saved to {filename}")
    else:
        print("The episode was not successful, GIF not saved.")

def show_agent(env, policy):
    """Shows the agent's behavior based on the learned policy"""
    state, _ = env.reset()  # corrected: only the initial state
    env.render()

    for t in range(1000):  # maximum number of steps
        action = policy[state]  # choose an action based on the policy
        next_state, reward, done, _, _ = env.step(action)  # corrected: 5 return values

        env.render()

        state = next_state  # update the state

        if done:
            break  # end if the episode is done

    env.close()

def generate_random_policy(env):
    """Generates a random policy"""
    policy = defaultdict(int)
    for state in range(env.observation_space.n):
        policy[state] = env.action_space.sample()
    return policy

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")
    env.reset(seed=42)

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.5

    num_episodes = 5000
    num_timesteps = 1000

    # Train using Q-learning
    Q = q_learning(env, num_episodes, num_timesteps, alpha, gamma, epsilon)

    # Generate a policy from the Q-function
    policy = generate_policy(Q, env)

    # Test the policy
    sum_reward = 0
    for _ in range(5000):
        total_reward = test(env, policy, render=False)
        sum_reward += total_reward

    print(f"Average reward over 5000 episodes: {sum_reward / 5000}")

    # Show successful completion in a GIF
    show_agent_gif(env, policy)

    env.close()
