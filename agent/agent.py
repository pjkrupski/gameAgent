import numpy as np


#Starter code via chatGPT
class SARSA_Agent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.Q_table = np.zeros((num_states, num_actions))  # Initialize Q-table with zeros
    
    def select_action(self, state):
        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            return np.random.choice(self.num_actions)
        else:
            # Exploitation: choose the action with the highest Q-value for the current state
            return np.argmax(self.Q_table[state])
    
    def update_Q_table(self, state, action, reward, next_state, next_action):
        # SARSA update rule
        current_Q_value = self.Q_table[state, action]
        next_Q_value = self.Q_table[next_state, next_action]
        td_target = reward + self.discount_factor * next_Q_value
        td_error = td_target - current_Q_value
        self.Q_table[state, action] += self.learning_rate * td_error
    
    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()  # Reset environment to start a new episode
            action = self.select_action(state)
            done = False
            
            while not done:
                next_state, reward, done, _ = env.step(action)  # Take an action and observe the next state and reward
                next_action = self.select_action(next_state)  # Choose next action
                self.update_Q_table(state, action, reward, next_state, next_action)  # Update Q-table
                state = next_state
                action = next_action
    
    def test(self, env, num_episodes):
        total_rewards = []
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = np.argmax(self.Q_table[state])  # Choose action with highest Q-value
                next_state, reward, done, _ = env.step(action)
                state = next_state
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        return total_rewards
