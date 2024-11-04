import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Fetch historical stock data
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')
prices = data['Close'].values

# Parameters
num_episodes = 1000
num_steps = len(prices) - 1
learning_rate = 0.1
discount_factor = 0.95
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01

# Action space: 0 - Hold, 1 - Buy, 2 - Sell
actions = [0, 1, 2]
q_table = np.zeros((num_steps, len(actions)))

# Function to calculate state features
def get_state(step):
    if step == 0:
        return 0, 0  # Initial state
    price_change = prices[step] - prices[step - 1]
    return price_change, prices[step]  # Price change and current price

# Training the agent
portfolio = []
for episode in range(num_episodes):
    current_position = 0  # 0: No stock, 1: Holding stock
    total_reward = 0

    for step in range(num_steps):
        # Get state features
        price_change, current_price = get_state(step)

        if np.random.rand() < epsilon:
            action = np.random.choice(actions)  # Explore
        else:
            action = np.argmax(q_table[step])  # Exploit

        if action == 1:  # Buy
            if current_position == 0:  # Can only buy if not holding
                current_position = 1
                total_reward -= current_price  # Cost of buying
        elif action == 2:  # Sell
            if current_position == 1:  # Can only sell if holding
                current_position = 0
                total_reward += current_price  # Gain from selling

        # Update Q-value
        next_state = step + 1 if step + 1 < num_steps else step
        price_change_next, _ = get_state(next_state)
        q_table[step][action] += learning_rate * (
            total_reward + discount_factor * np.max(q_table[next_state]) - q_table[step][action]
        )

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    portfolio.append(total_reward)

# Plotting the portfolio value over episodes
plt.plot(portfolio)
plt.xlabel('Episode')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Value over Episodes')
plt.show()
