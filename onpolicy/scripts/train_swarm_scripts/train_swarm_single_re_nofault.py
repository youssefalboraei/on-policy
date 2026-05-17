import os
import subprocess
import sys

# No-fault Reduced-env baseline.
# Same hyperparameters as best_cih (the global-10 / Standard-env training):
#   rmappo, episode_length=500, hidden=128, layer_N=3, stacked_frames=6,
#   ppo_epoch=10, lr=7e-4, critic_lr=1e-4, clip=0.2, entropy=0.015,
#   gain=0.01, data_chunk_length=32, n_rollout_threads=16, num_env_steps=200M.
# Only env-scale args differ: 3 agents, 3 boxes, 250x250 arena.
# Pairs with the same SwarmEnv.reset() edit on the nofault-baseline branch
# that removes the dynamic fault injection over {3,4,5,8}.

env = "SwarmEnv"
scenario = "single_transport_small"
num_agents = 3
nym_boxes = 3
arena_width = 250
arena_height = 250
algo = "rmappo"
exp = "re-nofault"
seed_max = 1

print(f"env is {env}, scenario is {scenario}, algo is {algo}, exp is {exp}, max seed is {seed_max}")

train_script_path = "train/train_swarm.py"

for seed in range(1, seed_max + 1):
    print(f"seed is {seed}:")

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    command = [
        sys.executable, train_script_path,
        # --cuda is store_false default=True; omit to use GPU.
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
        "--critic_lr", "0.002",
        "--wandb_name", "xxx",
        "--user_name", "ygalboraei-university-of-bristol",
        "--clip_param", "0.2",
        "--stacked_frames", "6",
        "--use_stacked_frames",
        "--hidden_size", "128",
        "--layer_N", "3",
        "--entropy_coef", "0.015",
        "--data_chunk_length", "30",
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
