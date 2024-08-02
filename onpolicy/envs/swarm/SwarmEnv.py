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
        self.agents = [f'agent_{i}' for i in range(self.num_agents)]
        self.max_num_agents = self.num_agents

        self.step_counter = 0
        np.random.seed(self.config.seed)
        # Define discrete action space for each agent
        self.action_space = {
            agent: spaces.Discrete(13)  # 21 discrete actions
            for agent in self.agents
        }
        
        # We'll define the observation space in reset() after creating the simulator
        self.reset()

    def reset(self, seed=None, fault_type=None, num_faults=None):
        if seed is not None:
            self.config.seed = seed
        seed = self.config.seed
        # np.random.seed(self.config.seed)
        self.config.seed = np.random.randint(0, 999999)

        if fault_type is not None:
            self.config.fault_type = fault_type

        if num_faults is not None:
            self.config.number_of_faults = number_of_faults

        self.simulator = marl_sim.FaultManagementSimulator(
            self.config,
            self.config.number_of_faults,
            self.config.fault_type,
            self.config.seed,
            self.config.predict_fault,
            # marl_sim.M_type.BY_MARL_SINGLE
            marl_sim.M_type.BY_MARL_SINGLE
        )
        # print("RESET------------")
        # print(self.config.number_of_faults, self.config.fault_type)
        # exit()

        self.previous_delivery_rate = self.simulator.bb.s_delivery_rate[-1]
        self.previous_rb_distance = np.array(self.simulator.bb.rb_distance).reshape(self.simulator.bb.s_no_robots, self.simulator.bb.s_no_boxes)
        self.previous_box_y_positions = np.array(self.simulator.bb.b_pos_y)
        self.base_box_y_positions = np.array(self.simulator.bb.b_pos_y)
        
        # Define the observation space based on the actual observation size
        sample_obs = self._get_observation()
        self.observation_space = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=sample_obs[agent].shape, dtype=np.float64)
            for agent in self.agents
        }
        # print(sample_obs)
        return sample_obs, {}

    def step(self, actions):
        # Apply actions by writing them directly to bb.r_mitigation_action
        mitigation_actions = [0.0] * self.num_agents  # Initialize with zeros
        for agent, action in actions.items():
            agent_index = int(agent.split('_')[1])
            # print(action)
            mitigation_actions[agent_index] = action[0]  
            # mitigation_actions[agent_index] = 0

        self.simulator.bb.r_mitigation_action = mitigation_actions
        
        self.simulator.step()  # Run the simulation step
        self.step_counter += 1
        # print("simulation done")
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        truncated = {agent: False for agent in self.agents}  # Assuming no truncation
        info = self._get_info()
        return observation, reward, done, truncated, info

    def _get_observation(self):
        bb = self.simulator.bb
        observations = {}
        for i, agent in enumerate(self.agents):
            agent_obs = np.array([
                bb.r_state[i],              ##
                # bb.r_positioned[i],
                # bb.r_in_team[i],
                # bb.r_lifting[i],
                bb.r_velocity_comp[i],      ##
                bb.r_nearest_box[i],        ##
                # bb.r_nearest_boxm[i],
                bb.r_nearest_robot[i],      ##
                bb.r_nearest_wall[i],       ##
                bb.r_nearest_box_id[i],
                # bb.r_nearest_boxm_id[i],
                bb.r_nearest_robot_id[i],
                bb.r_nearest_wall_id[i],
                bb.r_robots_in_range[i],    ##
                bb.r_box_in_range[i],       ##
                # bb.r_boxm_in_range[i],
                bb.r_walls_in_range[i],     ##
                bb.r_messages_r[i],
                bb.r_messages_s[i],
                bb.r_delivered[i],
                # bb.r_delivered_m[i]    
                
                
                # bb.r_robots_in_range[i],
                # bb.r_box_in_range[i],
                # bb.r_walls_in_range[i],
                # bb.r_velocity_comp[i],
                # bb.r_state[i],
                # bb.r_nearest_robot[i],
                # bb.r_nearest_box[i],
                # bb.r_nearest_wall[i]
            ], dtype=np.float64)
            agent_obs = np.nan_to_num(agent_obs, nan=0.0, posinf=0.0, neginf=0.0) #TODO: check the source of nan from the simulator
            observations[agent] = agent_obs
        # print(len(observations))
        # print(observations)
        return observations

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
        DELIVERY_REWARD = 10
        DISTANCE_TO_BOX_WEIGHT = 0.00
        # DISTANCE_TO_NEAREST_BOX_WEIGHT = 0.00
        DISTANCE_TO_DROP_AREA_WEIGHT = 10
        TIME_PENALTY = 0.1

        num_robots = self.simulator.bb.s_no_robots
        num_boxes = self.simulator.bb.s_no_boxes

        # Initialize rewards for each agent
        rewards = {agent: 0.0 for agent in self.agents}

        # 1. Reward for delivered boxes (global reward, split among agents)
        current_delivery_rate = self.simulator.bb.s_delivery_rate[-1]
        # print(current_delivery_rate)

        if hasattr(self, 'previous_delivery_rate'):
            # print('yes1')
            boxes_delivered = current_delivery_rate - self.previous_delivery_rate
            delivery_reward = DELIVERY_REWARD * boxes_delivered / num_robots
            for agent in self.agents:
                rewards[agent] += delivery_reward
        self.previous_delivery_rate = current_delivery_rate

        # 2. Reward for decreasing distance to boxes (per agent)
        rb_distance = np.array(self.simulator.bb.rb_distance).reshape(num_robots, num_boxes)
        # print(rbm_distance)

        if hasattr(self, 'previous_rbm_distance'):
            # print('yes2')
            for i, agent in enumerate(self.agents):
                distance_decrease = np.sum(self.previous_rb_distance[i] - rb_distance[i])
                rewards[agent] += DISTANCE_TO_BOX_WEIGHT * distance_decrease
        self.previous_rb_distance = rb_distance.copy()

        # 3. Reward for boxes moving towards drop area
        current_box_y_positions = np.array(self.simulator.bb.b_pos_y)
        # print(current_boxm_y_positions)

        if hasattr(self, 'previous_boxm_y_positions'):
            # print('yes3')
            y_progress = np.sum((current_box_y_positions - self.previous_box_y_positions) / (500.0 - self.base_box_y_positions))
            progress_reward = DISTANCE_TO_DROP_AREA_WEIGHT * y_progress / num_robots
            for agent in self.agents:
                rewards[agent] += progress_reward
        self.previous_boxm_y_positions = current_box_y_positions.copy()

        # 4. Time penalty (per agent)
        for agent in self.agents:
            rewards[agent] -= TIME_PENALTY

        return rewards



    def _is_done(self):

        done = self.simulator.completion_check() or (self.step_counter >= 2_000)
        if done:
            self.step_counter = 0
        return {agent: done for agent in self.agents}

    def _get_info(self):

        return {agent: {'delivery_rate': self.simulator.bb.s_delivery_rate[-1]}
                 for agent in self.agents}


    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            self.config.seed = seed
        return [self.config.seed]

    def observation_space(self, agent):
        return self.observation_space[agent]

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