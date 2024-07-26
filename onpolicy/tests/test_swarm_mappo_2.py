import torch
import numpy as np
import argparse
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.envs.swarm.swarm_env import SwarmEnvWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="SwarmEnv")
    parser.add_argument("--algorithm_name", type=str, default="mappo")
    parser.add_argument("--experiment_name", type=str, default="check")
    parser.add_argument('--scenario_name', type=str, default='single_transport', help="Which scenario to run on")
    parser.add_argument("--num_faults", type=int, default=0, help="number of faults")
    parser.add_argument('--fault_type', type=int, default=0, help="type of fault")
    parser.add_argument('--num_agents', type=int, default=10, help="number of players")
    parser.add_argument('--num_boxes', type=int, default=10, help="number of boxes")
    parser.add_argument('--num_mboxes', type=int, default=0, help="number of mboxes")
    parser.add_argument('--fault_number', type=int, default=0, help="number of faulty agents")
    parser.add_argument('--delivery_bias', type=int, default=1, help="delivery bias")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_rollout_threads", type=int, default=1)
    parser.add_argument("--episode_length", type=int, default=1500)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--layer_N", type=int, default=1)
    parser.add_argument("--use_ReLU", action="store_true")
    parser.add_argument("--use_popart", action="store_true")
    parser.add_argument("--use_valuenorm", action="store_true")
    parser.add_argument("--use_feature_normalization", action="store_true")
    parser.add_argument("--use_centralized_V", action="store_true")
    parser.add_argument("--use_obs_instead_of_state", action="store_true")
    parser.add_argument("--use_recurrent_policy", action="store_false")
    parser.add_argument("--recurrent_N", type=int, default=1)
    parser.add_argument("--data_chunk_length", type=int, default=10)
    parser.add_argument("--gain", type=float, default=0.01)
    parser.add_argument("--use_orthogonal", action="store_true")
    parser.add_argument("--use_policy_active_masks", action="store_true")
    
    return parser.parse_args()

def main(args):
    # Set up the environment
    env = SwarmEnvWrapper(args)
    env.seed(args.seed)
    
    # Set up the actor network
    actor = R_Actor(args, env.observation_space[0], env.action_space[0])
    
    # Load the trained model
    model_path = f"/actor.pt"
    try:
        actor.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    actor.eval()
    
    # Run test episodes
    num_episodes = 10
    total_reward = 0
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(args.episode_length):
            # Get action from the actor
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action, _, _ = actor(obs_tensor, None, None)
            
            # Convert action to numpy array
            action_np = action.squeeze(0).numpy()
            
            # Take a step in the environment
            obs, reward, done, info = env.step(action_np)
            episode_reward += np.mean(reward)
            episode_length += 1
            
            # Optional: Render if needed
            # env.render()
            
            if np.all(done):
                break
        
        total_reward += episode_reward
        print(f"Episode {episode + 1}: reward = {episode_reward:.2f}, length = {episode_length}")
    
    env.close()
    average_reward = total_reward / num_episodes
    print(f"Testing completed. Average reward over {num_episodes} episodes: {average_reward:.2f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)