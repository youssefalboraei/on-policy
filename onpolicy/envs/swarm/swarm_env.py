from onpolicy.envs.swarm.SwarmEnv import SwarmEnv
import numpy as np
import gym
from gym import spaces

class SwarmEnvWrapper(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, all_args):
        config_args = {
            "number_of_agents": all_args.num_agents,
            "seed": all_args.seed,
            # Add other parameters as needed
        }
        self.env = SwarmEnv(config_args)
        
        self.n_agents = self.env.num_agents
        self.agents = [f'agent_{i}' for i in range(self.n_agents)]
        
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        
        for agent in self.agents:
            self.action_space.append(self.env.action_space[agent])
            self.observation_space.append(self.env.observation_space[agent])
        
        # Assuming global state is the concatenation of all observations
        global_state_dim = sum([obs_space.shape[0] for obs_space in self.observation_space])
        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(global_state_dim,), dtype=np.float32) for _ in range(self.n_agents)]

        self.current_step = 0

    def step(self, actions):
        actions_dict = {agent: action for agent, action in zip(self.agents, actions)}
        obs, rew, done, truncated, info = self.env.step(actions_dict)
        
        obs_list = [obs[agent] for agent in self.agents]
        rew_list = [rew[agent] for agent in self.agents]
        done_list = [done[agent] for agent in self.agents]
        info_list = [{agent: info[agent]} for agent in self.agents]
        
        return np.array(obs_list), np.array(rew_list), np.array(done_list), info_list

    def reset(self):
        obs, _ = self.env.reset()
        self.current_step = 0
        return np.array([obs[agent] for agent in self.agents])

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def get_obs(self):
        return self.reset()

    def get_state(self):
        obs = self.get_obs()
        return np.concatenate(obs)

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.action_space[0].n))

    def get_env_info(self):
        return {
            "n_actions": self.action_space[0].n,
            "n_agents": self.n_agents,
            "state_shape": self.get_state().shape[0],
            "obs_shape": self.observation_space[0].shape[0],
            "episode_limit": 1000  # Set an appropriate episode limit
        }