import numpy as np
import pandas as pd
import os

class DataCollector2:
    RUN_NUMBER_FILE = 'last_run_number.txt'

    @classmethod
    def get_next_run_number(cls):
        if os.path.exists(cls.RUN_NUMBER_FILE):
            with open(cls.RUN_NUMBER_FILE, 'r') as f:
                last_run = int(f.read().strip())
        else:
            last_run = 0
        
        next_run = last_run + 1
        
        return next_run

    def __init__(self):
        self.data = []
        self.step = 0
        self.run_number = self.get_next_run_number()
        self.fault_number = None
        self.fault_type = None
        self.seed = None

    def set_run_info(self, fault_number, fault_type):
        self.fault_number = fault_number
        self.fault_type = fault_type

    def set_seed(self, seed):
        self.seed = seed

    def add_step_data(self, actions, delivery_rate):
        step_data = {
            'Step': self.step,
            'Delivery_Rate': delivery_rate
        }
        
        for i, action_index in enumerate(actions):
            step_data[f'Robot_{i}_Action'] = action_index
        
        self.data.append(step_data)
        self.step += 1

    def save_data(self):
        if self.fault_number is None or self.fault_type is None:
            raise ValueError("Fault info must be set before saving data.")
        if self.seed is None:
            raise ValueError("Seed must be set before saving data.")

        filename = f'simulation_data_run{self.run_number}_N{self.fault_number}T{self.fault_type}.csv'

        df = pd.DataFrame(self.data)
        df['Run'] = self.run_number
        df['Fault_Number'] = self.fault_number
        df['Fault_Type'] = self.fault_type
        df['Seed'] = self.seed

        # Reorder columns
        columns = ['Run', 'Fault_Number', 'Fault_Type', 'Seed', 'Step', 'Delivery_Rate'] + [col for col in df.columns if col.startswith('Robot_')]
        df = df[columns]

        df.to_csv(filename, index=False)
        print(f"Simulation data saved to {filename}")

        # Update the run number file after saving the data
        with open(self.RUN_NUMBER_FILE, 'w') as f:
            f.write(str(self.run_number))