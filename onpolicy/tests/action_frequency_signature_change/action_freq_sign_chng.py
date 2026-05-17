import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants
FAULT_NAMES = {
    0: ["NO_FAULT", "No fault"],
    1: ["SINGLE_WHEEL", "Single wheel"],
    2: ["DOUBLE_WHEEL", "Double wheel"],
    3: ["F1: ALL_WHEEL_V0", "F1: 0% speed"],
    4: ["F2: ALL_WHEEL_V10", "F2: 10% speed"],
    5: ["F3: ALL_WHEEL_V50", "F3: 50% speed"],
    6: ["RADIAL_CAM_4", "Radial cam"],
    7: ["UPFACING_CAM", "Upfacing cam"],
    8: ["F4: PICKUP", "F4: Pickup"],
    9: ["DROPOFF", "Dropoff"],
    10: ["LASER_16", "Laser"],
    11: ["R_COMMS", "R comms"],
    12: ["S_COMMS", "S comms"]
}

# ACTION_NAMES = [
#     "A1: NO_ACTION", "A2: DECREASE_SPEED_50", "A3: STOP_MOVING", "A4: BIAS_TO_NEAREST_ROBOT",
#     "A5: BIAS_TO_NEAREST_BOX", "A6: BIAS_TO_NEAREST_WALL", "A7: BIAS_LEFT",
#     "A8: BIAS_FROM_NEAREST_ROBOT", "A9: BIAS_FROM_NEAREST_BOX", "A10: BIAS_FROM_NEAREST_WALL",
#     "A11: ATTRACT_NEIGHBOUR", "A12: REPEL_NEIGHBOUR", "A13: DROP_BOX"
# ]

ACTION_NAMES = [
    "A1: No action", "A2: Decrease speed 50%", "A3: Stop moving", "A4: Bias to nearest robot",
    "A5: Bias to nearest box", "A6: Bias to nearest wall", "A7: Bias left",
    "A8: Bias from nearest robot", "A9: Bias from nearest box", "A10: Bias from nearest wall",
    "A11: Attract neighbour", "A12: Repel neighbour", "A13: Drop box"
]


def get_robot_counts(scenario):
    """Determine the total number of robots based on the scenario."""
    return 10 if 'SE' in scenario else 3

def process_simulation_file(file_path, scenario):
    """Process a single simulation file and average across all faulty and non-faulty robots."""
    df = pd.read_csv(file_path)
    
    # Extract number of faulty robots and fault type from filename
    num_faulty = int(file_path.split('_N')[1].split('T')[0])
    fault_type = int(file_path.split('_N')[1].split('T')[1].split('.')[0])
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
    
    return num_faulty, fault_type, faulty_counts, non_faulty_counts

def process_scenario_data(folder_path, scenario):
    """Process all simulation files in a scenario folder."""
    # Dictionary to store results for each N value
    n_results = {}
    
    for filename in os.listdir(folder_path):
        if filename.startswith('simulation_data_run'):
            file_path = os.path.join(folder_path, filename)
            num_faulty, fault_type, faulty_freqs, non_faulty_freqs = process_simulation_file(file_path, scenario)
            
            # Initialize results list for this N if not exists
            if num_faulty not in n_results:
                n_results[num_faulty] = []
            
            # Create row labels
            faulty_label = f"F T{fault_type}"
            non_faulty_label = f"NF T{fault_type}"
            
            # Process faulty robot data
            if faulty_freqs is not None:
                for action, freq in faulty_freqs.items():
                    n_results[num_faulty].append({
                        'Robot_State': faulty_label,
                        'Action': ACTION_NAMES[action],
                        'Frequency': freq * 100  # Convert to percentage
                    })
            
            # Process non-faulty robot data
            if non_faulty_freqs is not None:
                for action, freq in non_faulty_freqs.items():
                    n_results[num_faulty].append({
                        'Robot_State': non_faulty_label,
                        'Action': ACTION_NAMES[action],
                        'Frequency': freq * 100  # Convert to percentage
                    })
    
    # Convert each N's results to DataFrame and aggregate
    results_df = {}
    for n, results in n_results.items():
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.groupby(['Robot_State', 'Action'])['Frequency'].mean().reset_index()
        results_df[n] = df
    
    return results_df


def plot_3d_heatmap_by_fault(results_df, scenario, fault_type, robot_state='F'):
    """Generate 3D heatmap for a specific fault type showing action frequencies across different N values."""
    all_data = []
    for n, df in results_df.items():
        if not df.empty:
            # Filter for specific fault type and robot state
            state_data = df[df['Robot_State'].str.startswith(robot_state) & 
                          df['Robot_State'].str.endswith(f'T{fault_type}')]
            for _, row in state_data.iterrows():
                all_data.append({
                    'N': n,
                    'Action': row['Action'],
                    'Frequency': row['Frequency']
                })
    
    # Create DataFrame and pivot
    df_3d = pd.DataFrame(all_data)
    if df_3d.empty:
        print(f"No data for fault type {fault_type} in {scenario}")
        return
        
    pivot_df = df_3d.pivot(index='N', columns='Action', values='Frequency')
    pivot_df = pivot_df.reindex(columns=ACTION_NAMES).fillna(0)
    pivot_df = pivot_df.sort_index()
    
    plt.figure(figsize=(22, 12))
    
    # Plot heatmap
    ax = sns.heatmap(
        pivot_df,
        annot=True,
        cmap="Blues",
        vmin=0,
        vmax=100,
        fmt=".2f",
        annot_kws={"size": 16},
        cbar_kws={"label": "Frequency (%)"},
        linewidths=1,
        linecolor="white",
    )
    
    fault_desc = FAULT_NAMES[fault_type][1]
    plt.title(
        f"Action Frequencies vs Number of Faulty Robots - {scenario}\n{fault_desc} - {'Faulty' if robot_state == 'F' else 'Non-Faulty'} Robots",
        fontsize=18,
        pad=20,
    )
    plt.xlabel("Action", fontsize=16, labelpad=15)
    plt.ylabel("Number of Faulty Robots", fontsize=16)
    
    plt.xticks(fontsize=12, rotation=45, ha="right")
    plt.yticks(fontsize=12, rotation=0)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Frequency (%)", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(
        f"action_frequencies_3d_{scenario.lower()}_T{fault_type}_{robot_state}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()

def main():
    base_folder = '/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/tests'
    scenarios = ['RT-SE', 'ST-SE']
    fault_types = [3, 4, 5, 8]  # F1-F4
    
    for scenario in scenarios:
        folder_path = os.path.join(base_folder, scenario, 'with_mitigation')
        results_df = process_scenario_data(folder_path, scenario)
        
        for fault_type in fault_types:
            plot_3d_heatmap_by_fault(results_df, scenario, fault_type, 'F')
            plot_3d_heatmap_by_fault(results_df, scenario, fault_type, 'NF')

if __name__ == "__main__":
    main()