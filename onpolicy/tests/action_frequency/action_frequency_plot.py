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

def process_simulation_file(file_path):
    df = pd.read_csv(file_path)
    action_columns = [col for col in df.columns if col.startswith('Robot_') and col.endswith('_Action')]
    action_data = df[action_columns]
    
    action_counts = action_data.values.flatten().astype(int)
    action_counts = pd.Series(action_counts).value_counts()
    total_actions = action_counts.sum()
    action_frequencies = action_counts / total_actions
    
    return action_frequencies

def process_scenario_data(folder_path, scenario):
    results = []
    for filename in os.listdir(folder_path):
        if filename.startswith('simulation_data_run'):
            file_path = os.path.join(folder_path, filename)
            fault_type = int(filename.split('_N')[1].split('T')[1].split('.')[0])
            
            action_freqs = process_simulation_file(file_path)
            for action, freq in action_freqs.items():
                results.append({
                    'Scenario': scenario,
                    'Fault_Type': FAULT_NAMES[fault_type],
                    'Action': ACTION_NAMES[action],
                    'Frequency': freq
                })
    
    df = pd.DataFrame(results)
    
    # Aggregate data to handle duplicates
    df_agg = df.groupby(['Scenario', 'Fault_Type', 'Action'])['Frequency'].mean().reset_index()
    
    return df_agg

def plot_action_frequencies(df, scenario):
    plt.figure(figsize=(20, 12))
    sns.barplot(x='Fault_Type', y='Frequency', hue='Action', data=df)
    plt.title(f'Action Frequencies by Fault Type - {scenario}', fontsize=16)
    plt.xlabel('Fault Type', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Action', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'action_frequencies_{scenario.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

# def plot_action_heatmap(df, scenario):
    # plt.figure(figsize=(20, 10))
    
    # # Fill missing values with 0 in the pivot table
    # pivot_df = df.pivot(index='Fault_Type', columns='Action', values='Frequency').fillna(0)
    
    # ax = sns.heatmap(pivot_df, annot=True, cmap='Blues', vmin=0, fmt='.2f', annot_kws={'size': 20})
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=18) 
    
    # plt.title(f'Action Frequencies by Fault Type - {scenario}', fontsize=18)
    # plt.xlabel('Action', fontsize=18)
    # plt.ylabel('Fault Type', fontsize=18)
    
    # plt.xticks(fontsize = 16)
    # plt.yticks(fontsize = 16)

    # # Adjust x-axis (Action) labels
    # plt.xticks(rotation=45, ha='right')
    
    # # Adjust y-axis (Fault Type) labels
    # plt.yticks(rotation=0, va='center')
    
    # plt.tight_layout()
    # plt.savefig(f'action_frequencies_heatmap_{scenario.lower()}.png', dpi=300, bbox_inches='tight')
    # plt.close()

# def plot_action_heatmap(df, scenario):
#     plt.figure(figsize=(20, 10))
    
#     # Create the pivot table and fill missing values with 0
#     pivot_df = df.pivot(index='Fault_Type', columns='Action', values='Frequency').fillna(0)
    
#     # Reindex to ensure order follows FAULT_NAMES and ACTION_NAMES
#     fault_order = [FAULT_NAMES[i] for i in sorted(FAULT_NAMES.keys())]
#     action_order = ACTION_NAMES

#     # Reindex the pivot table
#     pivot_df = pivot_df.reindex(index=fault_order, columns=action_order)

#     # Filter out rows and columns with all zeros (no data)
#     pivot_df = pivot_df.loc[(pivot_df.sum(axis=1) != 0), (pivot_df.sum(axis=0) != 0)]

#     # Plot the heatmap
#     ax = sns.heatmap(pivot_df, annot=True, cmap='Blues', vmin=0, fmt='.2f', annot_kws={'size': 20})
#     cbar = ax.collections[0].colorbar
#     cbar.ax.tick_params(labelsize=18)
#     cbar.set_label('Frequency', fontsize=18)
    
#     plt.title(f'Action Frequencies by Fault Type - {scenario}', fontsize=18)
#     plt.xlabel('Action', fontsize=18)
#     plt.ylabel('Fault Type', fontsize=18)
    
#     plt.xticks(fontsize=16, rotation=45, ha='right')
#     plt.yticks(fontsize=16, rotation=0, va='center')
    
#     plt.tight_layout()
#     plt.savefig(f'action_frequencies_heatmap_{scenario.lower()}.png', dpi=300, bbox_inches='tight')
#     plt.close()


def plot_action_heatmap(df, scenario):
    plt.figure(figsize=(20, 10))
    
    # Create the pivot table and fill missing values with 0
    pivot_df = df.pivot(index='Fault_Type', columns='Action', values='Frequency').fillna(0)
    
    # Ensure order for actions and only include faults with data
    action_order = ACTION_NAMES
    faults_with_data = pivot_df.loc[(pivot_df.sum(axis=1) != 0)].index.tolist()
    
    # Reindex the pivot table
    pivot_df = pivot_df.reindex(index=faults_with_data, columns=action_order).fillna(0)

    # Plot the heatmap
    ax = sns.heatmap(pivot_df, annot=True, cmap='Blues', vmin=0, fmt='.2f', annot_kws={'size': 20})
    
    # Customize color bar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Frequency', fontsize=18)
    
    # Set plot labels and title
    plt.title(f'Action Frequencies by Fault Type - {scenario}', fontsize=18)
    plt.xlabel('Action', fontsize=18)
    plt.ylabel('Fault Type', fontsize=18)
    
    # Set x and y ticks formatting
    plt.xticks(fontsize=16, rotation=45, ha='right')
    plt.yticks(fontsize=16, rotation=0, va='center')
    
    # Adjust layout to fit all elements
    plt.tight_layout()
    plt.savefig(f'action_frequencies_heatmap_{scenario.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_stacked_bar_chart(df, scenario):
    plt.figure(figsize=(20, 12))
    pivot_df = df.pivot(index='Fault_Type', columns='Action', values='Frequency')
    pivot_df.plot(kind='bar', stacked=True)
    plt.title(f'Action Frequencies by Fault Type - {scenario}', fontsize=20)
    plt.xlabel('Fault Type', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.legend(title='Action', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()
    plt.savefig(f'action_frequencies_stacked_bar_{scenario.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    base_folder = '/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/tests'
    scenarios = ['RT-RE', 'RT-SE', 'ST-RE', 'ST-SE']
    
    for scenario in scenarios:
        folder_path = os.path.join(base_folder, scenario, 'with_mitigation')
        df = process_scenario_data(folder_path, scenario)
        
        # plot_action_frequencies(df, scenario)
        plot_action_heatmap(df, scenario)
        # plot_stacked_bar_chart(df, scenario)
        
        print(f"Processed and plotted data for {scenario}")

if __name__ == "__main__":
    main()
