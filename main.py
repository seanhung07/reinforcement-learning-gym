import numpy as np
import gym

env = gym.make('MountainCar-v0', render_mode="human")

# Q-table initialization
state_space_size = [20, 20]
q_table = np.random.uniform(low=-2, high=0, size=(state_space_size + [env.action_space.n]))

# Discretizing state space into bins
def discretize_state(state, bins):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    ratios = [(state[i] - env_low[i]) / (env_high[i] - env_low[i]) for i in range(len(state))]
    new_state = [int(round((bins[i] - 1) * ratios[i])) for i in range(len(bins))]
    new_state = [min(bins[i] - 1, max(0, new_state[i])) for i in range(len(bins))]
    return tuple(new_state)

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 5000  # Number of training episodes
epsilon_decay = 0.999  # Epsilon decay to reduce exploration over time

bins = [20, 20]

for episode in range(episodes):
    state, _ = env.reset() 
    state = discretize_state(state, bins)
    
    done = False
    total_reward = 0
    
    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, done, truncated, info = env.step(action)
        
        done = done or truncated
        
        next_state_discrete = discretize_state(next_state, bins)
        
        current_q = q_table[state + (action,)]
        max_future_q = np.max(q_table[next_state_discrete])
        new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
        q_table[state + (action,)] = new_q
        
        state = next_state_discrete
        total_reward += reward

    epsilon *= epsilon_decay
    if episode % 500 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

print("Training complete.")
