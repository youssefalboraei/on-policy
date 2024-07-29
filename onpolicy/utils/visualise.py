import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(action_filename='eval_data\eval_action_data.csv', delivery_filename='eval_data\eval_delivery_rate_data.csv'):
    action_data = pd.read_csv(action_filename)
    delivery_data = pd.read_csv(delivery_filename)
    return action_data, delivery_data

def visualize_action_frequencies(data):
    action_counts = data['Action'].value_counts()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=action_counts.index, y=action_counts.values)
    plt.title('Overall Action Frequencies')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    # plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('overall_action_frequency.png')
    print("Overall action frequency plot saved as 'overall_action_frequency.png'")

def visualize_action_distribution_over_time(data):
    pivot_data = pd.crosstab(data['Timestep'], data['Action'])
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_data, cmap='YlOrRd', cbar_kws={'label': 'Frequency'})
    plt.title('Action Distribution Over Time')
    plt.xlabel('Action')
    plt.ylabel('Timestep')
    plt.tight_layout()
    plt.savefig('action_distribution_over_time.png')
    print("Time distribution plot saved as 'action_distribution_over_time.png'")

def visualize_robot_action_distribution(data):
    pivot_data = pd.crosstab(data['Robot'], data['Action'])
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_data, cmap='YlOrRd', cbar_kws={'label': 'Frequency'})
    plt.title('Action Distribution Across Robots')
    plt.xlabel('Action')
    plt.ylabel('Robot')
    plt.tight_layout()
    plt.savefig('robot_action_distribution.png')
    print("Robot action distribution plot saved as 'robot_action_distribution.png'")

def visualize_delivery_rates_over_time(data):
    plt.figure(figsize=(12, 6))
    for robot in data['Robot'].unique():
        robot_data = data[data['Robot'] == robot]
        plt.plot(robot_data['Timestep'], robot_data['Delivery_Rate'], label=f'{robot}')
    
    plt.title('Delivery Rates Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Delivery Rate')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('delivery_rates_over_time.png')
    print("Delivery rates plot saved as 'delivery_rates_over_time.png'")

def visualize_average_delivery_rate(data):
    average_delivery_rate = data.groupby('Timestep')['Delivery_Rate'].mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(average_delivery_rate.index, average_delivery_rate.values)
    plt.title('Average Delivery Rate Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Average Delivery Rate')
    plt.tight_layout()
    plt.savefig('average_delivery_rate.png')
    print("Average delivery rate plot saved as 'average_delivery_rate.png'")

def visualize_delivery_rate_heatmap(data):
    pivot_data = data.pivot(index='Timestep', columns='Robot', values='Delivery_Rate')
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(pivot_data, cmap='YlOrRd', cbar_kws={'label': 'Delivery Rate'})
    plt.title('Delivery Rate Heatmap')
    plt.xlabel('Robot')
    plt.ylabel('Timestep')
    plt.tight_layout()
    plt.savefig('delivery_rate_heatmap.png')
    print("Delivery rate heatmap saved as 'delivery_rate_heatmap.png'")

def main():
    # Load the data
    action_data, delivery_data = load_data()

    # Generate visualizations for actions
    visualize_action_frequencies(action_data)
    visualize_action_distribution_over_time(action_data)
    visualize_robot_action_distribution(action_data)

    # Generate visualizations for delivery rates
    visualize_delivery_rates_over_time(delivery_data)
    visualize_average_delivery_rate(delivery_data)
    visualize_delivery_rate_heatmap(delivery_data)

if __name__ == "__main__":
    main()