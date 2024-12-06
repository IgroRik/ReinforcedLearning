import gym
import numpy as np
import imageio

def discretize_state(state, bins):
    """
    Converts continuous state to a discrete one.
    """
    state_bounds = [(-1.2, 0.6), (-0.07, 0.07)]  # boundaries for MountainCar
    state_bins = [
        np.linspace(bound[0], bound[1], bins[i]) for i, bound in enumerate(state_bounds)
    ]
    discretized_state = [
        np.digitize(state[i], state_bins[i]) - 1 for i in range(len(state))
    ]
    return tuple(discretized_state)

def compute_value_function(policy, env, state_space, num_iterations=1000, threshold=1e-20, gamma=0.99):
    """
    Computes the value function for a given policy.
    """
    value_table = np.zeros(state_space)

    for _ in range(num_iterations):
        updated_value_table = np.copy(value_table)

        for state in np.ndindex(state_space):
            position = state[0] * 0.01 - 1.2  # Convert discrete state to continuous
            velocity = state[1] * 0.0014 - 0.07
            current_state = np.array([position, velocity])

            action = policy[state]
            env.reset()
            env.unwrapped.state = current_state  # Set the environment's state
            next_state, reward, terminated, truncated, _ = env.step(action)

            next_state_discretized = discretize_state(next_state, state_space)
            done = terminated or truncated

            if done:
                value_table[state] = reward
            else:
                value_table[state] = reward + gamma * updated_value_table[next_state_discretized]

        if np.sum(np.abs(updated_value_table - value_table)) < threshold:
            break

    return value_table

def extract_policy(value_table, env, state_space, gamma=0.99):
    """
    Extracts the optimal policy based on the value function.
    """
    policy = np.zeros(state_space, dtype=int)

    for state in np.ndindex(state_space):
        position = state[0] * 0.01 - 1.2
        velocity = state[1] * 0.0014 - 0.07
        current_state = np.array([position, velocity])

        Q_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            env.reset()
            env.unwrapped.state = current_state
            next_state, reward, terminated, truncated, _ = env.step(action)

            next_state_discretized = discretize_state(next_state, state_space)
            done = terminated or truncated

            if done:
                Q_values[action] = reward
            else:
                Q_values[action] = reward + gamma * value_table[next_state_discretized]

        policy[state] = np.argmax(Q_values)

    return policy

import gym
import imageio

def save_training_gif(env, policy, bins, filename="Lab2/trainingPolicy.gif"):
    """
    Saves the policy execution process as a gif.
    """
    images = []
    state = discretize_state(env.reset()[0], bins)
    done = False

    while not done:
        img = env.render()  # In the new version of gym, render() returns a frame if render_mode="rgb_array"
        images.append(img)
        action = policy[state]
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = discretize_state(next_state, bins)
        done = terminated or truncated

    env.close()
    imageio.mimsave(filename, images, fps=30)
    print(f"Gif saved as {filename}")

def policy_iteration(env, state_space):
    """
    Implements Policy Iteration for the discretized space.
    """
    policy = np.zeros(state_space, dtype=int)

    for _ in range(1000):
        value_function = compute_value_function(policy, env, state_space)
        new_policy = extract_policy(value_function, env, state_space)

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return policy

def test(env, optimal_policy, state_space, render=True):
    """
    Tests the environment using the optimal policy.
    """
    state = discretize_state(env.reset(seed=42)[0], state_space)
    total_reward = 0

    if render:
        env.render()

    for _ in range(1000):
        action = optimal_policy[state]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = discretize_state(next_state, state_space)

        if render:
            env.render()

        if done:
            break

    return total_reward

if __name__ == "__main__":
    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    bins = (36, 28)  # number of discrete states for each axis
    state_space = (bins[0], bins[1])
    
    optimal_policy = policy_iteration(env, state_space)
    save_training_gif(env, optimal_policy, bins)
    print("Optimal policy found!")

    # Testing
    total_rewards = []
    for _ in range(10):
        total_reward = test(env, optimal_policy, state_space, render=False)
        total_rewards.append(total_reward)
    
    print("Average reward:", np.mean(total_rewards))
