import numpy as np
import pandas as pd

class DataCollector:
    def __init__(self):
        self.action_names = [
            "NO_ACTION", "DECREASE_SPEED_50", "STOP_MOVING", "BIAS_TO_NEAREST_ROBOT",
            "BIAS_TO_NEAREST_BOX", "BIAS_TO_NEAREST_WALL", "BIAS_LEFT",
            "BIAS_FROM_NEAREST_ROBOT", "BIAS_FROM_NEAREST_BOX", "BIAS_FROM_NEAREST_WALL",
            "ATTRACT_NEIGHBOUR", "REPEL_NEIGHBOUR", "DROP_BOX"
        ]
        self.action_history = []
        self.delivery_rate_history = []
        self.timestep = 0

    def add_actions(self, actions):
        actions = np.squeeze(actions)  # Remove the first dimension if it's 1
        num_robots = actions.shape[0]
        
        action_indices = np.argmax(actions, axis=1)
        action_names = [self.action_names[i] for i in action_indices]
        
        action_data = pd.DataFrame({
            'Timestep': self.timestep,
            'Robot': range(num_robots),
            'Action': action_names
        })
        
        self.action_history.append(action_data)
        self.timestep += 1

    def add_delivery_rates(self, eval_infos):
        delivery_data = []
        for agent_data in eval_infos[0]:
            for agent, data in agent_data.items():
                delivery_data.append({
                    'Timestep': self.timestep - 1,  # Subtract 1 because we've already incremented timestep in add_actions
                    'Robot': agent,
                    'Delivery_Rate': data['delivery_rate']
                })
        
        self.delivery_rate_history.append(pd.DataFrame(delivery_data))

    def save_data(self, action_filename='action_data.csv', delivery_filename='delivery_rate_data.csv'):
        full_action_data = pd.concat(self.action_history, ignore_index=True)
        full_action_data.to_csv(action_filename, index=False)
        print(f"Action data saved to {action_filename}")

        full_delivery_data = pd.concat(self.delivery_rate_history, ignore_index=True)
        full_delivery_data.to_csv(delivery_filename, index=False)
        print(f"Delivery rate data saved to {delivery_filename}")
