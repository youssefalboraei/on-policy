from onpolicy.envs.swarm.SwarmEnv import SwarmEnv

import numpy as np

def test_swarm_env():
    # Configuration for the environment
    config_args = {
        "a": 3, "b": 3, "ft": 3, "fn": 2, "aw": 250, "ah": 250,# "-mid", "0",
        "bm": 0, "i": 10_000, "s": 53395, "-dbias": 1
    }

    # Create the environment
    env = SwarmEnv(config_args)
    print("env created")

    # Print environment information
    print(f"Number of agents: {env.num_agents}")
    print("Action space:")
    for agent, space in env.action_space.items():
        print(f"  {agent}: {space}")
    print("Observation space:")
    for agent, space in env.observation_space.items():
        print(f"  {agent}: {space}")

    # Run a few steps to inspect the environment behavior
    obs, _ = env.reset()
    print("\nInitial observation:")
    for agent, o in obs.items():
        print(f"  {agent}:")
        for i, value in enumerate(o):
            print(f"    Attribute {i}: {value}")

    for step in range(10_000):  # Run for 5 steps
        # actions = {agent: env.action_space[agent].sample() for agent in env.agents}
        actions = {agent: [0] for agent in env.agents}
        
        # print("Actions selected:")
        # print(f"  {actions}")
        
        next_obs, rewards, dones, truncated, info = env.step(actions)

        # print(next_obs['agent_0'][-3])
        # print(next_obs)
        print(step)
        
        # print(f"\nAfter step {step + 1}:")
        # print("Actions applied:")
        # print(f"  {env.simulator.bb.r_mitigation_action}")
        # print("Observations:")
        # for agent, o in next_obs.items():
        #     print(f"  {agent}:")
        #     for i, value in enumerate(o):
        #         print(f"    Attribute {i}: {value}")

        # Attribute names
        # attributes = [
        #     "r_robots_in_range",
        #     "r_box_in_range",
        #     "r_walls_in_range",
        #     "r_velocity_comp",
        #     "r_state",
        #     "r_nearest_robot",
        #     "r_nearest_box",
        #     "r_nearest_wall"
        # ]

        # print("Observations:")

        # # Extract the agents and their observations
        # agents = list(next_obs.keys())
        # observations = list(next_obs.values())

        # # Print headers
        # header = " " * 18 + "  ".join(f"{agent:<10}" for agent in agents)
        # print(header)

        # # Print each attribute row by row
        # for i, attribute in enumerate(attributes):
        #     row = f"{attribute:<18}:"
        #     for o in observations:
        #         value = f"{o[i]:<10.2f}" if i < len(o) else 'N/A'
        #         row += f"    {value}"
        #     print(row)
        # exit()
        # print(f"Rewards: {rewards}")
        # print(f"Dones: {dones}")
        # print(f"Info: {info}")

        # print(dones.values())
        # print(info)

        if all(dones.values()):
            print("All done")
            break

    env.close()

if __name__ == "__main__":
    test_swarm_env()