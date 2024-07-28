import marl_sim
import numpy as np
import gym
from gym import spaces
import argparse

class SwarmEnv(gym.Env):
    def __init__(self, config_args):
        super(SwarmEnv, self).__init__()
        self.config = marl_sim.Config(self.dict_to_config_list(config_args))
        
        self.config.compute_delivery_rate = True
        self.config.compute_metrics = True #
        self.config.predict_fault = False
        self.simulator = None
        self.num_agents = self.config.number_of_agents
        self.num_boxes = self.config.number_of_boxes
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.max_num_agents = self.num_agents

        self.step_counter = 0

        # Define discrete action space for each agent
        self.action_space = {
            agent: spaces.Discrete(15)  # 21 discrete actions
            for agent in self.agents
        }
        
        # We'll define the observation space in reset() after creating the simulator
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.config.seed = seed
        
        np.random.seed(self.config.seed)
        self.config.seed = np.random.randint(0, 999999)

        # seed = self.config.seed
        # print(seed)
        self.step_counter = 0
        self.simulator = marl_sim.FaultManagementSimulator(
            self.config,
            self.config.number_of_faults,
            self.config.fault_type,
            self.config.seed,
            self.config.predict_fault,
            # marl_sim.M_type.BY_MARL_SINGLE
            marl_sim.M_type.BY_MARL_SINGLE
        )
        self.previous_delivery_rate = self.simulator.bb.s_delivery_rate[-1]
        # self.previous_rb_distance = np.array(self.simulator.bb.rb_distance).reshape(self.simulator.bb.s_no_robots, self.simulator.bb.s_no_boxes)
        self.previous_nearest_box_distances = np.array(self.simulator.bb.r_nearest_box)
        self.previous_box_y_positions = np.array(self.simulator.bb.b_pos_y)
        
        # Define the observation space based on the actual observation size
        sample_obs = self._get_observation()
        sample_share_obs = self._get_share_observation()

        self.observation_space = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs[agent].shape, dtype=np.float64)
            for agent in self.agents
        }

        # global_state_dim = (sample_obs[agent].shape * self.num_agents) + (2 * self.num_agents) + (2 * self.num_boxes) # 2 is for  x and y
        # self.share_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=global_state_dim , dtype=np.float64)
        self.share_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=sample_share_obs.shape , dtype=np.float64)
        

        # print(sample_obs)
        return sample_obs, sample_share_obs, {}

    def step(self, actions):
        self.step_counter += 1
        # Apply actions by writing them directly to bb.r_mitigation_action
        mitigation_actions = [0.0] * self.num_agents  # Initialize with zeros
        for agent, action in actions.items():
            agent_index = int(agent.split('_')[1])
            # print(action)
            mitigation_actions[agent_index] = action[0] 
            # mitigation_actions[agent_index] = 0

        self.simulator.bb.r_mitigation_action = mitigation_actions
        
        self.simulator.step()  # Run the simulation step
        # print("simulation done")
        observation = self._get_observation()
        share_observation = self._get_share_observation()
        reward = self._get_reward()
        done = self._is_done()
        truncated = {agent: False for agent in self.agents}  # Assuming no truncation
        info = self._get_info()            
        
        return observation, share_observation, reward, done, truncated, info


    def _get_observation(self):
        bb = self.simulator.bb
        observations = {}
        for i, agent in enumerate(self.agents):
            agent_obs = np.array([
                bb.r_state[i],
                # bb.r_positioned[i],
                # bb.r_in_team[i],
                # bb.r_lifting[i],
                bb.r_velocity_comp[i],
                bb.r_nearest_box[i],
                # bb.r_nearest_boxm[i],
                bb.r_nearest_robot[i],
                bb.r_nearest_wall[i],
                bb.r_nearest_box_id[i],
                # bb.r_nearest_boxm_id[i],
                bb.r_nearest_robot_id[i],
                bb.r_nearest_wall_id[i],
                bb.r_robots_in_range[i],
                bb.r_box_in_range[i],
                # bb.r_boxm_in_range[i],
                bb.r_walls_in_range[i],
                bb.r_messages_r[i],
                bb.r_messages_s[i],
                bb.r_delivered[i],
                # bb.r_delivered_m[i]
            ], dtype=np.float32)
            observations[agent] = agent_obs
        return observations

    def _get_share_observation(self):
        bb = self.simulator.bb
        observations = self._get_observation()
        
        # Flatten individual observations into a single array
        flattened_obs = np.concatenate([obs for obs in observations.values()])
        
        # Add global state information
        global_state = np.concatenate([
            bb.r_pos_x,
            bb.r_pos_y,
            bb.b_pos_x,
            bb.b_pos_y
        ])
        
        # Combine flattened observations and global state
        share_observations = np.concatenate([flattened_obs, global_state])
        return share_observations

    # def _get_reward(self):
    #     # Implement reward calculation for each agent
    #     # This is a placeholder and should be replaced with your actual reward logic
    #     self.reward = self.simulator.bb.s_delivery_rate_m[-1]
    #     if self.reward > self.prev_reward:
    #         return {agent: self.reward for agent in self.agents}
    #     else:
    #         return {agent: 0 for agent in self.agents}


    def _get_reward(self):
        # Constants
        DELIVERY_REWARD = 500.0
        # DISTANCE_TO_BOX_WEIGHT = 0.00
        DISTANCE_TO_NEAREST_BOX_WEIGHT = 0.00
        DISTANCE_TO_DROP_AREA_WEIGHT = 0.1
        TIME_PENALTY = 0.01

        num_robots = self.simulator.bb.s_no_robots
        num_boxes = self.simulator.bb.s_no_boxes

        # Initialize rewards for each agent
        rewards = {agent: 0.0 for agent in self.agents}

        # 1. Reward for delivered boxes (global reward, split among agents)
        current_delivery_rate = self.simulator.bb.s_delivery_rate[-1]

        if hasattr(self, 'previous_delivery_rate'):
            boxes_delivered = current_delivery_rate - self.previous_delivery_rate
            delivery_reward = DELIVERY_REWARD * boxes_delivered #/ num_robots
            for agent in self.agents:
                rewards[agent] += delivery_reward
        self.previous_delivery_rate = current_delivery_rate

        # # 2. Reward for decreasing distance to boxes (per agent)
        # rb_distance = np.array(self.simulator.bb.rb_distance).reshape(num_robots, num_boxes)
        # exit()

        # if hasattr(self, 'previous_rbm_distance'):
        #     for i, agent in enumerate(self.agents):
        #         distance_decrease = np.sum(self.previous_rb_distance[i] - rb_distance[i])
        #         rewards[agent] += DISTANCE_TO_BOX_WEIGHT * distance_decrease
        # self.previous_rb_distance = rb_distance.copy()

        # 2. Reward for decreasing distance to nearest box
        current_nearest_box_distances = np.array(self.simulator.bb.r_nearest_box)
        
        if hasattr(self, 'previous_nearest_box_distances'):
            for i, agent in enumerate(self.agents):
                distance_decrease = self.previous_nearest_box_distances[i] - current_nearest_box_distances[i]
                rewards[agent] += DISTANCE_TO_NEAREST_BOX_WEIGHT * distance_decrease
    
        self.previous_nearest_box_distances = current_nearest_box_distances.copy()

        # 3. Reward for boxes moving towards drop area
        current_box_y_positions = np.array(self.simulator.bb.b_pos_y)

        if hasattr(self, 'previous_boxm_y_positions'):
            y_progress = np.sum(current_box_y_positions - self.previous_box_y_positions)
            progress_reward = DISTANCE_TO_DROP_AREA_WEIGHT * y_progress / num_robots
            for agent in self.agents:
                rewards[agent] += progress_reward
        self.previous_box_y_positions = current_box_y_positions.copy()

        # 4. Time penalty (per agent)
        for agent in self.agents:
            rewards[agent] -= TIME_PENALTY

        # print(rewards)
        return rewards


    def _is_done(self):
        # Implement your termination condition
        # For example, you could end the episode after a fixed number of steps
        done = self.simulator.completion_check() or (self.step_counter >= 1_500)
        return {agent: done for agent in self.agents}

    def _get_info(self):
        # Return any additional information you want to log
        return {agent: {'delivery_rate': self.simulator.bb.s_delivery_rate[-1]}
                 for agent in self.agents}
        # return {
        #     "delivery_rate": self.simulator.bb.s_delivery_rate,
        #     # "delivery_rate_m": self.simulator.bb.s_delivery_rate_m
        # }

    def render(self, mode='human'):
        # Implement rendering if needed
        pass

    def close(self):
        # Implement any cleanup here if needed
        pass

    def seed(self, seed=None):
        if seed is not None:
            self.config.seed = seed
        return [self.config.seed]

    def observation_space(self, agent):
        return self.observation_space[agent]

    def share_observation_space(self):
        return self.share_observation_space

    def action_space(self, agent):
        return self.action_space[agent]
    
    def clean_data(data):
        for key, value in data.items():
            data[key] = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        return data
    
    def clean_list(data):
        arr = np.array(data)
        cleaned_arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return cleaned_arr.tolist()
    
    def dict_to_config_list(self, config_dict):
        config_list = []
        for key, value in config_dict.items():
            config_list.append(f"-{key}")
            config_list.append(str(value))
        return config_list