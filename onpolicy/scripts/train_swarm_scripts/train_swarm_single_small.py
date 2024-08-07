import os
import subprocess

env = "SwarmEnv"
scenario = "single_transport_small"
num_agents = 3
nym_boxes = 3
arena_width = 250
arena_height = 250
algo = "rmappo"  # "rmappo" "ippo"
exp = "small"
seed_max = 1

print(f"env is {env}, scenario is {scenario}, algo is {algo}, exp is {exp}, max seed is {seed_max}")

# Specify the full path to train_mpe.py
train_script_path = r"D:\youssef\on-policy\onpolicy\scripts\train\train_swarm.py"
    
for seed in range(1, seed_max + 1):
    print(f"seed is {seed}:")
    
    # Set environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Construct the command
    command = [
        "python", train_script_path,
        "--cuda", "False", #
        "--env_name", env,
        "--algorithm_name", algo,
        "--experiment_name", exp,
        "--scenario_name", scenario,
        "--delivery_bias", "1",
        "--num_agents", str(num_agents),
        "--num_boxes", str(nym_boxes),
        "--arena_width", str(arena_width),
        "--arena_height", str(arena_height),
        "--seed", str(seed),
        "--n_training_threads", "1",
        "--n_eval_rollout_threads", "1",
        "--n_rollout_threads", "16", #
        "--num_mini_batch", "1",
        "--episode_length", "1_000",
        "--num_env_steps", "200_000_000", #
        "--ppo_epoch", "10", # 10
        "--use_ReLU",
        "--gain", "0.01",
        "--lr", "7e-4",
        "--critic_lr", "1e-3", #, "7e-4"
        "--wandb_name", "xxx",
        "--user_name", "ygalboraei-university-of-bristol",

        "--clip_param", "0.2",
        "--stacked_frames", "4", # 4 
        "--use_stacked_frames",
        "--hidden_size", "128", # 512
        "--layer_N", "2",
        "--entropy_coef", "0.015",
        "--data_chunk_length", "20",

        "--num_faults", "1",
        "--fault_type", "8", 
    ]

    #      echo "seed is ${seed}:"
    # CUDA_VISIBLE_DEVICES=1 python eval/eval_hanabi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    # --hanabi_name ${hanabi} --num_agents ${num_agents} --seed 1 --n_training_threads 128 --n_rollout_threads 1 \
    # --n_eval_rollout_threads 1000 --num_mini_batch 4 --episode_length 100 --num_env_steps 10000000000000 --ppo_epoch 15 \
    # --gain 0.01 --lr 7e-4 --critic_lr 1e-3 --hidden_size 512 --layer_N 2 --use_eval --use_wandb --use_recurrent_policy \
    # --entropy_coef 0.015 --model_dir "xxx"
    # echo "training is done!"
    
    # Run the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")