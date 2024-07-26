import torch
import numpy as np
import argparse
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.envs.swarm.swarm_env import SwarmEnvWrapper
from onpolicy.config import get_config
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="SwarmEnv")
    parser.add_argument("--algorithm_name", type=str, default="mappo")
    parser.add_argument("--experiment_name", type=str, default="check")
    parser.add_argument('--scenario_name', type=str,
                        default='single_transport', help="Which scenario to run on")
    parser.add_argument("--num_fualts", type=int, default=0, 
                        help="number of faults")
    parser.add_argument('--fault_type', type=int,
                        default=0, help="type of fault")
    parser.add_argument('--num_agents', type=int,
                        default=10, help="number of players")
    parser.add_argument('--num_boxes', type=int,
                        default=10, help="number of boxes")
    parser.add_argument('--num_mboxes', type=int,
                        default=0, help="number of mboxes")
    parser.add_argument('--fault_number', type=int,
                        default=0, help="number of faulty agents")
    parser.add_argument('--delivery_bias', type=int,
                        default=1, help="delivery bias")
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
    
    return parser.parse_args()

def make_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = SwarmEnvWrapper(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return SubprocVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def main(args):
    # Set up the environment
    env = make_env(args)
    
    # Get environment info
    env_info = env.get_env_info()
    
    # Set up the actor and critic networks
    actor = R_Actor(args, env.observation_space[0], env.action_space[0])
    # critic = R_Critic(args, env.observation_space[0])
    
    # Load the trained model
    model_path = f"tests\actor.pt"
    actor.load_state_dict(torch.load(model_path))
    actor.eval()
    
    # Run test episodes
    num_episodes = 10
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        if args.use_recurrent_policy:
            rnn_states = torch.zeros(1, args.num_agents, args.recurrent_N, args.hidden_size)
        else:
            rnn_states = None
        
        for step in range(args.episode_length):
            # Get action from the actor
            with torch.no_grad():
                if args.use_recurrent_policy:
                    action, _, rnn_states = actor(torch.FloatTensor(obs).unsqueeze(0), 
                                                  rnn_states, 
                                                  torch.ones(1, args.num_agents, 1))
                else:
                    action, _, _ = actor(torch.FloatTensor(obs).unsqueeze(0), 
                                         torch.zeros(1, args.num_agents, args.hidden_size), 
                                         torch.ones(1, args.num_agents, 1))
            
            # Take a step in the environment
            obs, reward, done, _ = env.step(action.squeeze().numpy())
            episode_reward += np.mean(reward)
            
            # Render if needed
            env.render()
            
            if np.all(done):
                break
        
        print(f"Episode {episode + 1} reward: {episode_reward}")
    
    env.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)