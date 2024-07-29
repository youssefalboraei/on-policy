import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ActionTracker:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.action_history = []

    def add_actions(self, actions):
        # Assuming actions is a 2D array where each row represents a robot's action
        self.action_history.append(actions)

    def get_cumulative_frequencies(self):
        all_actions = np.concatenate(self.action_history)
        action_counts = np.sum(all_actions, axis=0)
        return action_counts

    def visualize_action_frequencies(self):
        cumulative_freq = self.get_cumulative_frequencies()
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(self.num_actions), cumulative_freq)
        plt.title('Cumulative Action Frequencies Across All Robots and Timesteps')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.xticks(range(self.num_actions), [f'Action_{i}' for i in range(self.num_actions)])
        plt.tight_layout()
        plt.savefig('cumulative_action_frequency.png')
        print("Cumulative plot saved as 'cumulative_action_frequency.png'")

        # Save the data
        pd.DataFrame({'Action': [f'Action_{i}' for i in range(self.num_actions)],
                      'Frequency': cumulative_freq}).to_csv('cumulative_action_frequency.csv', index=False)
        print("Cumulative data saved as 'cumulative_action_frequency.csv'")

    def visualize_action_distribution_over_time(self):
        action_history_array = np.array(self.action_history)
        action_frequencies = np.sum(action_history_array, axis=1)  # Sum over robots for each timestep
        
        plt.figure(figsize=(12, 6))
        plt.imshow(action_frequencies.T, aspect='auto', interpolation='nearest')
        plt.title('Action Distribution Over Time')
        plt.xlabel('Timestep')
        plt.ylabel('Action')
        plt.colorbar(label='Frequency')
        plt.yticks(range(self.num_actions), [f'Action_{i}' for i in range(self.num_actions)])
        plt.tight_layout()
        plt.savefig('action_distribution_over_time.png')
        print("Time distribution plot saved as 'action_distribution_over_time.png'")
