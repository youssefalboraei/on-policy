import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants
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

def get_robot_counts(scenario):
    """Determine the total number of robots based on the scenario."""
    return 10 if 'SE' in scenario else 3

def process_simulation_file(file_path, scenario):
    """Process a single simulation file and separate faulty and non-faulty robot actions."""
    df = pd.read_csv(file_path)
    
    # Extract number of faulty robots from filename
    num_faulty = int(file_path.split('_N')[1].split('T')[0])
    total_robots = get_robot_counts(scenario)
    
    # Get all action columns
    action_columns = [col for col in df.columns if col.startswith('Robot_') and col.endswith('_Action')]
    
    # Separate faulty and non-faulty robot actions
    faulty_columns = action_columns[:num_faulty] if num_faulty > 0 else []
    non_faulty_columns = action_columns[num_faulty:] if num_faulty < total_robots else []
    
    # Process faulty robots
    faulty_counts = None
    if faulty_columns:
        faulty_actions = df[faulty_columns].values.flatten().astype(int)
        faulty_counts = pd.Series(faulty_actions).value_counts()
        faulty_counts = faulty_counts / len(faulty_actions)
    
    # Process non-faulty robots
    non_faulty_counts = None
    if non_faulty_columns:
        non_faulty_actions = df[non_faulty_columns].values.flatten().astype(int)
        non_faulty_counts = pd.Series(non_faulty_actions).value_counts()
        non_faulty_counts = non_faulty_counts / len(non_faulty_actions)
    
    return faulty_counts, non_faulty_counts

def process_scenario_data(folder_path, scenario):
    """Process all simulation files in a scenario folder."""
    faulty_results = []
    non_faulty_results = []
    
    for filename in os.listdir(folder_path):
        if filename.startswith('simulation_data_run'):
            file_path = os.path.join(folder_path, filename)
            fault_type = int(filename.split('_N')[1].split('T')[1].split('.')[0])
            
            faulty_freqs, non_faulty_freqs = process_simulation_file(file_path, scenario)
            
            # Process faulty robot data
            if faulty_freqs is not None:
                for action, freq in faulty_freqs.items():
                    faulty_results.append({
                        'Scenario': scenario,
                        'Fault_Type': FAULT_NAMES[fault_type],
                        'Action': ACTION_NAMES[action],
                        'Frequency': freq,
                        'Robot_Type': 'Faulty'
                    })
            
            # Process non-faulty robot data
            if non_faulty_freqs is not None:
                for action, freq in non_faulty_freqs.items():
                    non_faulty_results.append({
                        'Scenario': scenario,
                        'Fault_Type': FAULT_NAMES[fault_type],
                        'Action': ACTION_NAMES[action],
                        'Frequency': freq,
                        'Robot_Type': 'Non-Faulty'
                    })
    
    # Combine and aggregate results
    df_faulty = pd.DataFrame(faulty_results) if faulty_results else None
    df_non_faulty = pd.DataFrame(non_faulty_results) if non_faulty_results else None
    
    # Aggregate data to handle duplicates
    if df_faulty is not None:
        df_faulty = df_faulty.groupby(['Scenario', 'Fault_Type', 'Action', 'Robot_Type'])['Frequency'].mean().reset_index()
    if df_non_faulty is not None:
        df_non_faulty = df_non_faulty.groupby(['Scenario', 'Fault_Type', 'Action', 'Robot_Type'])['Frequency'].mean().reset_index()
    
    return df_faulty, df_non_faulty

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
    
    # Adjust label formatting
    plt.xticks(fontsize=14, rotation=45, ha='right')
    plt.yticks(fontsize=16, rotation=0, va='center')
    
    # Adjust layout to remove white space
    plt.tight_layout()
    
    # Additional adjustments to remove extra spacing
    plt.subplots_adjust(bottom=0.2)
    
    plt.savefig(f'action_frequencies_{scenario.lower()}_{robot_type.lower()}_T{fault_type}.png', 
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def main():
    base_folder = '/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/tests'
    scenarios = ['RT-RE', 'RT-SE', 'ST-RE', 'ST-SE']
    
    for scenario in scenarios:
        folder_path = os.path.join(base_folder, scenario, 'with_mitigation')
        df_faulty, df_non_faulty = process_scenario_data(folder_path, scenario)
        
        # Generate separate heatmaps for faulty and non-faulty robots
        plot_action_heatmap(df_faulty, scenario, 'Faulty')
        plot_action_heatmap(df_non_faulty, scenario, 'Non-Faulty')
        
        print(f"Processed and plotted data for {scenario}")

if __name__ == "__main__":
    main()
