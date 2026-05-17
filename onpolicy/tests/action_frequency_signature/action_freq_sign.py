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
    """Process all simulation files in a scenario folder or load from cache if available."""
    # Check if cached data exists
    cache_file = f'frequency_data_{scenario.lower()}.csv'
    if os.path.exists(cache_file):
        print(f"Loading cached data for {scenario}")
        return pd.read_csv(cache_file)
    
    results = []
    for filename in os.listdir(folder_path):
        if filename.startswith('simulation_data_run'):
            file_path = os.path.join(folder_path, filename)
            fault_type = int(filename.split('_N')[1].split('T')[1].split('.')[0])
            
            faulty_freqs, non_faulty_freqs = process_simulation_file(file_path, scenario)
            
            # Create row label for both robot types (without dashes)
            faulty_label = f"F"  # Removed T{fault_type}
            non_faulty_label = f"NF"  # Removed T{fault_type}
            
            # Process faulty robot data
            if faulty_freqs is not None:
                for action, freq in faulty_freqs.items():
                    results.append({
                        'Robot_State': faulty_label,
                        'Action': ACTION_NAMES[action],
                        'Frequency': freq * 100  # Convert to percentage
                    })
            
            # Process non-faulty robot data
            if non_faulty_freqs is not None:
                for action, freq in non_faulty_freqs.items():
                    results.append({
                        'Robot_State': non_faulty_label,
                        'Action': ACTION_NAMES[action],
                        'Frequency': freq * 100  # Convert to percentage
                    })
    
    # Create DataFrame and aggregate
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.groupby(['Robot_State', 'Action'])['Frequency'].mean().reset_index()
        # Cache the processed data
        df.to_csv(cache_file, index=False)
        print(f"Cached data saved for {scenario}")
    
    return df

def process_scenario_data(folder_path, scenario):
    """Process all simulation files in a scenario folder or load from cache if available."""
    # Check if cached data exists
    cache_file = f'frequency_data_{scenario.lower()}.csv'
    if os.path.exists(cache_file):
        print(f"Loading cached data for {scenario}")
        df = pd.read_csv(cache_file)
        # Clean up any dashes in loaded data
        df['Robot_State'] = df['Robot_State'].str.replace(' - ', '')
        df['Robot_State'] = df['Robot_State'].str.replace('-', '')
        return df
    
    results = []
    for filename in os.listdir(folder_path):
        if filename.startswith('simulation_data_run'):
            file_path = os.path.join(folder_path, filename)
            fault_type = int(filename.split('_N')[1].split('T')[1].split('.')[0])
            
            faulty_freqs, non_faulty_freqs = process_simulation_file(file_path, scenario)
            
            # Create row label for both robot types (without dashes)
            faulty_label = f"F T{fault_type}"  # No dash
            non_faulty_label = f"NF T{fault_type}"  # No dash
            
            # Process faulty robot data
            if faulty_freqs is not None:
                for action, freq in faulty_freqs.items():
                    results.append({
                        'Robot_State': faulty_label,
                        'Action': ACTION_NAMES[action],
                        'Frequency': freq * 100  # Convert to percentage
                    })
            
            # Process non-faulty robot data
            if non_faulty_freqs is not None:
                for action, freq in non_faulty_freqs.items():
                    results.append({
                        'Robot_State': non_faulty_label,
                        'Action': ACTION_NAMES[action],
                        'Frequency': freq * 100  # Convert to percentage
                    })
    
    # Create DataFrame and aggregate
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.groupby(['Robot_State', 'Action'])['Frequency'].mean().reset_index()
        # Cache the processed data
        df.to_csv(cache_file, index=False)
        print(f"Cached data saved for {scenario}")
    
    return df

def plot_combined_heatmap(df, scenario):
    """Generate heatmap showing faulty and non-faulty robots together with real spacing."""
    if df.empty:
        print(f"No data available for scenario {scenario}")
        return
    
    plt.figure(figsize=(22, 14))
    
    # Create pivot table
    pivot_df = df.pivot(index="Robot_State", columns="Action", values="Frequency")
    pivot_df = pivot_df.reindex(columns=ACTION_NAMES).fillna(0)
    
    # Get fault types
    fault_types = sorted(set(int(idx.split("T")[1]) for idx in pivot_df.index))
    
    # Create new index with combined labels and spacing
    new_rows = []
    row_data = []
    
    # Fault type descriptions
    fault_descriptions = {
        0: "No fault     ",
        3: "F1: 0% speed     ",
        4: "F2: 10% speed     ",
        5: "F3: 50% speed    ",
        8: "F4: Can't pickup     ",
    }
    
    # Special case for no fault (only NF row)
    if 0 in fault_types:
        new_rows.append("No fault")
        nf_idx = f"NF T0"
        if nf_idx in pivot_df.index:
            row_data.append(pivot_df.loc[nf_idx].values)
        new_rows.append("")  # Add spacing
        row_data.append([np.nan] * len(ACTION_NAMES))
    
    # Process other fault types
    for i, fault_type in enumerate([ft for ft in fault_types if ft != 0]):
        # Add F row with fault name
        f_idx = f"F T{fault_type}"
        if f_idx in pivot_df.index:
            new_rows.append(f"{fault_descriptions.get(fault_type, f'Fault type {fault_type}')} F")
            row_data.append(pivot_df.loc[f_idx].values)
        
        # Add NF row
        nf_idx = f"NF T{fault_type}"
        if nf_idx in pivot_df.index:
            new_rows.append("NF")
            row_data.append(pivot_df.loc[nf_idx].values)
        
        # Add spacing after each pair except the last
        if i < len([ft for ft in fault_types if ft != 0]) - 1:
            new_rows.append("")
            row_data.append([np.nan] * len(ACTION_NAMES))
    
    # Create new DataFrame with spacers
    new_df = pd.DataFrame(row_data, index=new_rows, columns=pivot_df.columns)
    
    # Plot heatmap
    ax = sns.heatmap(
        new_df,
        annot=True,
        cmap="Blues",
        vmin=0,
        vmax=100,
        fmt=".2f",
        annot_kws={"size": 25},
        cbar_kws={"label": "Frequency (%)"},
        mask=new_df.isna(),
        linewidths=2,
        linecolor="white",
    )

    
    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    cbar.set_label("Frequency (%)", fontsize=25)
    
    # Set plot labels and title
    plt.title(
        f"{scenario}",
        fontsize=25,
        pad=20,
    )
    plt.xlabel("Action", fontsize=25, labelpad=15)
    
    # Adjust tick labels
    plt.xticks(fontsize=23, rotation=45, ha="right")
    
    # Set y-axis labels
    ax.set_yticklabels(new_rows, rotation=0, fontsize=25)
    
    # Hide labels for spacing rows
    yticklabels = ax.get_yticklabels()
    for label in yticklabels:
        if label.get_text() == "":
            label.set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(
        f"action_frequencies_combined_{scenario.lower()}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()




def main():
    base_folder = '/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/tests'
    scenarios = ['RT-SE', 'ST-SE']
    
    for scenario in scenarios:
        folder_path = os.path.join(base_folder, scenario, 'with_mitigation')
        df = process_scenario_data(folder_path, scenario)
        plot_combined_heatmap(df, scenario)
        print(f"Processed and plotted data for {scenario}")

if __name__ == "__main__":
    main()