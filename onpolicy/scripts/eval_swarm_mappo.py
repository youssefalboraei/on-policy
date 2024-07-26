import subprocess
import os

def run_evaluation(env, num_agents, num_boxes, algo, exp, scenario, seed):
    train_script_path = r"\\nstu-nas01.uwe.ac.uk\users4$\y2-alboraei\Windows\Downloads\on-policy\onpolicy\scripts\eval\eval_swarm.py"

    cmd = [
        "python", train_script_path,
        "--env_name", env,
        "--cuda", "False",
        "--algorithm_name", algo,
        "--experiment_name", exp,
        "--scenario_name", scenario,
        "--num_agents", str(num_agents),
        "--num_boxes", str(num_boxes),
        "--seed", str(seed),
        "--n_training_threads", "1",
        "--n_rollout_threads", "1",
        "--n_eval_rollout_threads", "1",  # Consider increasing this if you have more computational resources
        "--num_mini_batch", "1",
        "--episode_length", "30_000",
        "--num_env_steps", "150000",
        "--ppo_epoch", "10",
        "--use_ReLU",
        "--gain", "0.01",
        "--lr", "7e-4",
        "--critic_lr", "7e-4",
        "--wandb_name", "SwarmEnv_eval",
        "--user_name", "ygalboraei-university-of-bristol",
        "--use_eval",
        "--use_wandb",
        "--model_dir", r"\\nstu-nas01.uwe.ac.uk\users4$\y2-alboraei\Windows\Downloads\on-policy\onpolicy\tests"
    ]

    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        subprocess.run(cmd, env=env_vars, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running evaluation: {e}")

def main():
    env = "SwarmEnv"
    num_agents = 10
    num_boxes = 10
    algo = "mappo"
    exp = "check"
    scenario = "single_transport"
    seed_max = 1

    print(f"env is {env}, algo is {algo}, exp is {exp}, scenario is {scenario}, max seed is {seed_max}")

    for seed in range(1, seed_max + 1):
        print(f"seed is {seed+5}:")
        run_evaluation(env, num_agents, num_boxes, algo, exp, scenario, seed+5)
        print("evaluation is done!")

if __name__ == "__main__":
    main()