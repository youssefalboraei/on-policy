import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants (keeping existing constants)
FAULT_NAMES = {
    0: "NO_FAULT",
    1: "SINGLE_WHEEL",
    2: "DOUBLE_WHEEL",
    3: "F1: ALL_WHEEL_V0",
    4: "F2: ALL_WHEEL_V10",
    5: "F3: ALL_WHEEL_V50",
    6: "RADIAL_CAM_4",
    7: "UPFACING_CAM",
    8: "F4: PICKUP",
    9: "DROPOFF",
    10: "LASER_16",
    11: "R_COMMS",
    12: "S_COMMS"
}

ACTION_NAMES = [
    "A1: NO_ACTION", "A2: DECREASE_SPEED_50", "A3: STOP_MOVING", "A4: BIAS_TO_NEAREST_ROBOT",
    "A5: BIAS_TO_NEAREST_BOX", "A6: BIAS_TO_NEAREST_WALL", "A7: BIAS_LEFT",
    "A8: BIAS_FROM_NEAREST_ROBOT", "A9: BIAS_FROM_NEAREST_BOX", "A10: BIAS_FROM_NEAREST_WALL",
    "A11: ATTRACT_NEIGHBOUR", "A12: REPEL_NEIGHBOUR", "A13: DROP_BOX"
]

def process_simulation_file(file_path, num_faulty_robots):
    df = pd.read_csv(file_path)
    action_columns = [col for col in df.columns if col.startswith('Robot_') and col.endswith('_Action')]
    total_robots = len(action_columns)
    
    # Initialize empty Series for both robot types
    faulty_frequencies = pd.Series(dtype=float)
    non_faulty_frequencies = pd.Series(dtype=float)
    
    if num_faulty_robots == 0:
        # Case 1: No faulty robots - all robots are non-faulty
        non_faulty_data = df[action_columns]
        non_faulty_counts = non_faulty_data.values.flatten().astype(int)
        non_faulty_counts = pd.Series(non_faulty_counts).value_counts()
        non_faulty_total = non_faulty_counts.sum()
        non_faulty_frequencies = non_faulty_counts / non_faulty_total if non_faulty_total > 0 else pd.Series()
    
    elif num_faulty_robots == total_robots:
        # Case 2: All robots are faulty
        faulty_data = df[action_columns]
        faulty_counts = faulty_data.values.flatten().astype(int)
        faulty_counts = pd.Series(faulty_counts).value_counts()
        faulty_total = faulty_counts.sum()
        faulty_frequencies = faulty_counts / faulty_total if faulty_total > 0 else pd.Series()
    
    else:
        # Case 3: Mix of faulty and non-faulty robots
        faulty_columns = action_columns[:num_faulty_robots]
        non_faulty_columns = action_columns[num_faulty_robots:]
        
        # Process faulty robots
        if faulty_columns:
            faulty_data = df[faulty_columns]
            faulty_counts = faulty_data.values.flatten().astype(int)
            faulty_counts = pd.Series(faulty_counts).value_counts()
            faulty_total = faulty_counts.sum()
            faulty_frequencies = faulty_counts / faulty_total if faulty_total > 0 else pd.Series()
        
        # Process non-faulty robots
        if non_faulty_columns:
            non_faulty_data = df[non_faulty_columns]
            non_faulty_counts = non_faulty_data.values.flatten().astype(int)
            non_faulty_counts = pd.Series(non_faulty_counts).value_counts()
            non_faulty_total = non_faulty_counts.sum()
            non_faulty_frequencies = non_faulty_counts / non_faulty_total if non_faulty_total > 0 else pd.Series()
    
    return faulty_frequencies, non_faulty_frequencies

def process_scenario_data(folder_path, scenario):
    # Dictionary to store results, organized by fault type
    results = {
        'faulty': {},
        'non_faulty': {}
    }
    
    for filename in os.listdir(folder_path):
        if filename.startswith('simulation_data_run'):
            file_path = os.path.join(folder_path, filename)
            
            # Extract fault information from filename
            # New parsing logic for filenames like "simulation_data_run1000_N3T3.csv"
            try:
                # Split at underscore to get the last part containing N and T values
                n_t_part = filename.split('_')[-1]  # Gets "N3T3.csv"
                # Split this part to separate N and T values
                n_value = n_t_part.split('T')[0][1:]  # Gets "3" from "N3"
                t_value = n_t_part.split('T')[1].split('.')[0]  # Gets "3" from "T3.csv"
                
                num_faulty_robots = int(n_value)
                fault_type = int(t_value)
            except (IndexError, ValueError) as e:
                print(f"Error parsing filename {filename}: {e}")
                continue
            
            faulty_freqs, non_faulty_freqs = process_simulation_file(file_path, num_faulty_robots)
            
            # Initialize fault type dictionaries if they don't exist
            if fault_type not in results['faulty']:
                results['faulty'][fault_type] = []
            if fault_type not in results['non_faulty']:
                results['non_faulty'][fault_type] = []
            
            # Store results with N value
            if not faulty_freqs.empty and num_faulty_robots > 0:
                for action, freq in faulty_freqs.items():
                    results['faulty'][fault_type].append({
                        'Scenario': scenario,
                        'N_Faults': num_faulty_robots,
                        'Action': ACTION_NAMES[action],
                        'Frequency': freq
                    })
            
            if not non_faulty_freqs.empty and num_faulty_robots < len([col for col in pd.read_csv(file_path).columns if col.startswith('Robot_') and col.endswith('_Action')]):
                for action, freq in non_faulty_freqs.items():
                    results['non_faulty'][fault_type].append({
                        'Scenario': scenario,
                        'N_Faults': num_faulty_robots,
                        'Action': ACTION_NAMES[action],
                        'Frequency': freq
                    })
    
    # Convert to DataFrames
    processed_results = {
        'faulty': {},
        'non_faulty': {}
    }
    
    for robot_type in ['faulty', 'non_faulty']:
        for fault_type in results[robot_type]:
            if results[robot_type][fault_type]:  # Check if there's data
                df = pd.DataFrame(results[robot_type][fault_type])
                # Aggregate data to handle duplicates
                df = df.groupby(['Scenario', 'N_Faults', 'Action'])['Frequency'].mean().reset_index()
                processed_results[robot_type][fault_type] = df
    
    return processed_results


def plot_action_heatmap(df, scenario, robot_type, fault_type):
    if df.empty:
        print(f"No data available for {robot_type} robots in scenario {scenario} with fault type {fault_type}")
        return
    
    plt.figure(figsize=(22, 10))
    
    # Create the pivot table
    pivot_df = df.pivot(index='N_Faults', columns='Action', values='Frequency')
    
    # Ensure ALL actions are present by reindexing with the full ACTION_NAMES list
    pivot_df = pivot_df.reindex(columns=ACTION_NAMES)
    
    # Fill NaN values with 0 BEFORE sorting
    pivot_df = pivot_df.fillna(0)
    
    # Sort index by number of faults
    pivot_df = pivot_df.sort_index()
    
    # Plot the heatmap with adjusted parameters
    ax = sns.heatmap(pivot_df, 
                     annot=True, 
                     cmap='Blues', 
                     vmin=0, 
                     fmt='.2f', 
                     annot_kws={'size': 20},
                     cbar_kws={'label': 'Frequency'})
    
    # Customize color bar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Frequency', fontsize=18)
    
    # Set plot labels and title
    plt.title(f'Action Frequencies vs N Faults - {scenario}\n{robot_type} Robots, {FAULT_NAMES[fault_type]}', 
              fontsize=18, pad=20)
    plt.xlabel('Action', fontsize=18, labelpad=15)
    plt.ylabel('Number of Faulty Robots', fontsize=18)
    
    # Increased font sizes for axis labels
    plt.xticks(fontsize=16, rotation=45, ha='right')  # Increased from 14
    plt.yticks(fontsize=18, rotation=0, va='center')  # Increased from 16
    
    # Get the axes object
    ax = plt.gca()
    
    # Increase font size of x-axis (action names) tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    
    # Increase font size of y-axis (number of faults) tick labels
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
    
    # Adjust layout to remove white space
    plt.tight_layout()
    
    # Additional adjustments to remove extra spacing and accommodate larger fonts
    plt.subplots_adjust(bottom=0.25)  # Increased from 0.2 to accommodate larger font
    
    plt.savefig(f'action_frequencies_{scenario.lower()}_{robot_type.lower()}_T{fault_type}.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def main():
    base_folder = '/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/tests'
    scenarios = ['RT-RE', 'RT-SE', 'ST-RE', 'ST-SE']
    
    for scenario in scenarios:
        folder_path = os.path.join(base_folder, scenario, 'with_mitigation')
        results = process_scenario_data(folder_path, scenario)
        
        # Generate heatmaps for each fault type
        for robot_type in ['faulty', 'non_faulty']:
            for fault_type in results[robot_type]:
                df = results[robot_type][fault_type]
                plot_action_heatmap(df, scenario, robot_type.replace('_', ' ').title(), fault_type)
        
        print(f"Processed and plotted data for {scenario}")

if __name__ == "__main__":
    main()
