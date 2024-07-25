import os
import subprocess

env = "Swarm"
scenario = "collective-transport"
num_landmarks = 3
num_agents = 10
algo = "rmappo"  # "mappo" "ippo"
exp = "check"
seed_max = 1

print(f"env is {env}, scenario is {scenario}, algo is {algo}, exp is {exp}, max seed is {seed_max}")

# Specify the full path to train_mpe.py
train_script_path = r"\\nstu-nas01.uwe.ac.uk\users4$\y2-alboraei\Windows\Downloads\on-policy\onpolicy\scripts\train\train_swarm.py"

for seed in range(1, seed_max + 1):
    print(f"seed is {seed}:")
    
    # Set environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Construct the command
    command = [
        "python", train_script_path,
        "--cuda", "False",
        "--env_name", env,
        "--algorithm_name", algo,
        "--experiment_name", exp,
        "--scenario_name", scenario,
        "--num_agents", str(num_agents),
        "--num_landmarks", str(num_landmarks),
        "--seed", str(seed),
        "--n_training_threads", "1",
        "--n_rollout_threads", "128",
        "--num_mini_batch", "1",
        "--episode_length", "25",
        "--num_env_steps", "20000000",
        "--ppo_epoch", "10",
        "--use_ReLU",
        "--gain", "0.01",
        "--lr", "7e-4",
        "--critic_lr", "7e-4",
        "--wandb_name", "xxx",
        "--user_name", "ygalboraei-university-of-bristol"
    ]
    
    # Run the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")