import gym
import numpy as np
import time
import imageio

def discretize_state(state, bins):
    """
    Converts a continuous state into a discretized index.
    """
    state_idx = []
    for i in range(len(state)):
        state_idx.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_idx)

def create_bins(env, bins_per_dimension):
    """
    Creates discretized bins for each dimension of the state.
    """
    obs_space_low = env.observation_space.low
    obs_space_high = env.observation_space.high
    bins = []
    for dim in range(len(obs_space_low)):
        bins.append(np.linspace(obs_space_low[dim], obs_space_high[dim], bins_per_dimension[dim] - 1))
    return bins

def value_iteration(env, bins, num_iterations=1000, threshold=1e-4, gamma=0.99):
    """
    Implements value iteration for MountainCar-v0.
    """
    state_space_shape = tuple(len(bin_edges) + 1 for bin_edges in bins)
    value_table = np.zeros(state_space_shape)

    for iteration in range(num_iterations):
        updated_value_table = np.copy(value_table)

        for state_indices in np.ndindex(value_table.shape):
            state = np.array([bins[i][state_indices[i]] if state_indices[i] < len(bins[i]) else bins[i][-1]
                              for i in range(len(bins))])
            state = state + 0.001  # small shift to avoid hitting the bin boundary

            Q_values = []
            for action in range(env.action_space.n):
                # First, reset the environment to get the new state
                state_reset = env.reset()
                next_state, reward, terminated, truncated, info = env.step(action)  # modified here

                if terminated or truncated:
                    Q_values.append(reward)
                else:
                    next_state_idx = discretize_state(next_state, bins)
                    Q_values.append(reward + gamma * updated_value_table[next_state_idx])
                env.reset()  # Reset the environment after each action

            updated_value_table[state_indices] = max(Q_values)

        if np.max(np.abs(updated_value_table - value_table)) < threshold:
            print(f"Value iteration completed in {iteration} iterations.")
            break

        value_table = updated_value_table

    return value_table

def extract_policy(env, value_table, bins):
    gamma = 1.0
    policy = np.zeros_like(value_table)

    # Traverse all states (or discretized states)
    for state_indices in np.ndindex(value_table.shape):
        state = np.array([bins[i][state_indices[i]] if state_indices[i] < len(bins[i]) else bins[i][-1]
                          for i in range(len(bins))])

        Q_values = []
        for action in range(env.action_space.n):
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                Q_values.append(reward)
            else:
                next_state_idx = discretize_state(next_state, bins)
                Q_values.append(reward + gamma * value_table[next_state_idx])

        policy[state_indices] = np.argmax(Q_values)

    return policy

def save_training_gif(env, policy, bins, filename="Lab2/trainingValue.gif"):
    frames = []
    state = env.reset()
    done = False
    
    while not done:
        # Convert state to index if it is a tuple
        # If the state is multi-dimensional, extract the first part (e.g., position) for the index
        if isinstance(state, tuple):
            state = state[0]  # Extract the first element from the tuple (state)
        
        state_index = discretize_state(state, bins)  # Convert state to index
        
        # Get action from the policy for the current state
        action = int(policy[state_index])  # Convert to an integer (discrete action)
        
        # Step in the environment with the chosen action
        state, reward, done, truncated, info = env.step(action)
        
        # Capture frame for GIF
        frame = env.render()  # Get the frame in RGB format
        frames.append(frame)
        
        # Handle episode completion
        if done or truncated:
            break
    
    # Save frames as a GIF
    imageio.mimsave(filename, frames, duration=0.5)
    print(f"GIF saved to {filename}")

def test(env, optimal_policy, bins, render=False):
    """
    Tests the environment using the optimal policy.
    """
    state = env.reset()
    state_idx = discretize_state(state[0], bins)  # Here state[0] is the first value of the state
    total_reward = 0

    done = False
    while not done:
        if render:
            env.render()

        action = int(optimal_policy[state_idx])  # Convert to an integer (discrete action)

        state, reward, terminated, truncated, info = env.step(action)  # Now unpacking 5 values
        state_idx = discretize_state(state, bins)
        total_reward += reward

        # Check if the episode has finished
        if terminated or truncated:
            done = True

    env.close()
    return total_reward

if __name__ == '__main__':
    env = gym.make('MountainCar-v0', render_mode="rgb_array")
    
    # Example of binning the state
    position_bins = np.linspace(-1.2, 0.6, 24)
    velocity_bins = np.linspace(-0.07, 0.07, 24)
    bins = [position_bins, velocity_bins]

    optimal_value_function = value_iteration(env=env, bins=bins)
    optimal_policy = extract_policy(env, optimal_value_function, bins)

    print("Optimal policy:", optimal_policy)

    # Testing with GIF rendering
    save_training_gif(env, optimal_policy, bins)

    total_reward = test(env, optimal_policy, bins)
    print(f"Total reward: {total_reward}")

    sum_reward = 0
    for _ in range(5000):
        total_reward = test(env, optimal_policy, bins, render=False)
        sum_reward += total_reward

    print(f"Average reward: {sum_reward / 5000}")

    env.close()
