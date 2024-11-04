import numpy as np
import matplotlib.pyplot as plt
import random

# Define the grid environment
class GridEnvironment:
    def __init__(self, grid_size=(5, 5), goal_position=(4, 4), obstacles=[(1, 1), (2, 2), (3, 1)]):
        self.grid_size = grid_size
        self.goal_position = goal_position
        self.obstacles = obstacles
        self.state = (0, 0)  # Start position
    
    def reset(self):
        """Reset environment to the starting state."""
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        """Take an action in the environment and return next state, reward, done."""
        x, y = self.state

        # Define actions: 0=Up, 1=Right, 2=Down, 3=Left
        if action == 0:  # Move up
            x = max(0, x - 1)
        elif action == 1:  # Move right
            y = min(self.grid_size[1] - 1, y + 1)
        elif action == 2:  # Move down
            x = min(self.grid_size[0] - 1, x + 1)
        elif action == 3:  # Move left
            y = max(0, y - 1)

        new_state = (x, y)
        
        # Check for obstacles
        if new_state in self.obstacles:
            reward = -10  # Negative reward for hitting an obstacle
            done = False
        elif new_state == self.goal_position:
            reward = 100  # Positive reward for reaching the goal
            done = True
        else:
            reward = -1  # Small penalty for each step taken
            done = False

        self.state = new_state
        return new_state, reward, done

    def action_space(self):
        """Return the number of possible actions."""
        return 4  # Up, Right, Down, Left
    
    def state_space(self):
        """Return the state space dimensions."""
        return self.grid_size[0] * self.grid_size[1]

# Initialize environment
env = GridEnvironment()

# Q-learning parameters
num_episodes = 500
learning_rate = 0.1
discount_factor = 0.95
epsilon = 1.0
epsilon_decay = 0.99
epsilon_min = 0.01

# Initialize the Q-table
q_table = np.zeros((env.grid_size[0], env.grid_size[1], env.action_space()))

# Training loop
rewards = []
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        x, y = state

        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, env.action_space() - 1)  # Explore
        else:
            action = np.argmax(q_table[x, y])  # Exploit best action

        # Take action
        next_state, reward, done = env.step(action)
        total_reward += reward
        nx, ny = next_state

        # Update Q-value
        q_table[x, y, action] = q_table[x, y, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[nx, ny]) - q_table[x, y, action]
        )

        # Move to the next state
        state = next_state

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

# Visualize training progress
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Rewards over Episodes')
plt.show()

# Testing the trained agent
state = env.reset()
done = False
path = [state]  # To store the path taken by the agent

while not done:
    x, y = state
    action = np.argmax(q_table[x, y])  # Choose the best action based on learned Q-table
    state, _, done = env.step(action)
    path.append(state)

# Plot the grid and path taken by the agent
grid = np.zeros(env.grid_size)
for obs in env.obstacles:
    grid[obs] = -1  # Mark obstacles
grid[env.goal_position] = 1  # Mark goal

plt.imshow(grid, cmap='cool', origin='upper')
path_x, path_y = zip(*path)
plt.plot(path_y, path_x, marker='o', color='g', label='Path')
plt.scatter(env.goal_position[1], env.goal_position[0], color='red', marker='*', s=200, label='Goal')
plt.legend()
plt.title('Agent Path in Grid Environment')
plt.show()
