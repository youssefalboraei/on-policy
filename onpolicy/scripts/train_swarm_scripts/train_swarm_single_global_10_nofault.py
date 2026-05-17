import os
import subprocess
import sys

# No-fault Standard-env baseline.
# Mirrors train_swarm_single_global_10.py byte-for-byte except experiment_name.
# Pairs with the SwarmEnv.reset() edit on the nofault-baseline branch that
# removes the dynamic fault injection (np.random.randint over {3,4,5,8}).
# Produces a checkpoint comparable to best_a99 (ST) but trained without faults.

env = "SwarmEnv"
scenario = "single_transport_small"
num_agents = 10
nym_boxes = 10
arena_width = 500
arena_height = 500
algo = "rmappo"  # "rmappo" "ippo"
exp = "global-10-nofault"
seed_max = 1

print(f"env is {env}, scenario is {scenario}, algo is {algo}, exp is {exp}, max seed is {seed_max}")

train_script_path = "train/train_swarm.py"

for seed in range(1, seed_max + 1):
    print(f"seed is {seed}:")

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    command = [
        sys.executable, train_script_path,
        # --cuda is store_false default=True; omit to use GPU (passing it would DISABLE cuda).
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
        "--n_rollout_threads", "16",
        "--num_mini_batch", "1",
        "--episode_length", "500",
        "--num_env_steps", "200_000_000",
        "--ppo_epoch", "10",
        "--use_ReLU",
        "--gain", "0.01",
        "--lr", "7e-4",
        "--critic_lr", "1e-4",
        "--wandb_name", "xxx",
        "--user_name", "ygalboraei-university-of-bristol",
        "--clip_param", "0.2",
        "--stacked_frames", "6",
        "--use_stacked_frames",
        "--hidden_size", "128",
        "--layer_N", "3",
        "--entropy_coef", "0.015",
        "--data_chunk_length", "32",
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
