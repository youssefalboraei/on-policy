import subprocess
import os
from random import randint
import numpy as np

def run_evaluation(env, num_agents, num_boxes, algo, exp, scenario, seed, fault_type, num_faults):
    # train_script_path = "/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/scripts/eval/eval_swarm.py"
    train_script_path = r"D:\youssef\on-policy\onpolicy\scripts\eval\eval_swarm.py"

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
        "--arena_height", "250",
        "--arena_width", "250",
        "--delivery_bias", "1",
        "--n_training_threads", "1",
        "--n_rollout_threads", "1",
        "--n_eval_rollout_threads", "1",  # Consider increasing this if you have more computational resources
        "--num_mini_batch", "1",
        "--episode_length", "1_000",
        "--num_env_steps", "16_000",
        "--ppo_epoch", "10",
        "--use_ReLU",
        "--gain", "0.01",
        "--lr", "7e-4",
        "--critic_lr", "7e-4",
        "--wandb_name", "SwarmEnv_eval",
        "--user_name", "ygalboraei-university-of-bristol",
        "--use_eval",
        "--use_wandb",
        "--stacked_frames", "4", # 4 
        "--use_stacked_frames",
        "--hidden_size", "128", # 512
        "--layer_N", "2",
        # "--use_naive_recurrent_policy",
        # "--model_dir", "/home/yga/MSc_Robotics/Dissertation/on-policy/onpolicy/tests",
        "--model_dir", r"D:\youssef\on-policy\onpolicy\tests",
        "--num_faults", str(num_faults),
        "--fault_type", str(fault_type)
    ]

    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        subprocess.run(cmd, env=env_vars, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running evaluation: {e}")

def main():
    env = "SwarmEnv"
    num_agents = 3
    num_boxes = 3
    algo = "rmappo"
    exp = "check"
    scenario = "single_transport"
    seed_max = 1
    np.random.seed(0)

    print(f"env is {env}, algo is {algo}, exp is {exp}, scenario is {scenario}, max seed is {seed_max}")

    # for seed in range(1, seed_max + 1):
    #     seed = randint(1, 999)
    #     # seed = 704
    #     # seed = 706
    #     # seed = 707
    #     # seed = 709
    #     seed = 708
    #     print(f"seed is {seed}:")
    #     run_evaluation(env, num_agents, num_boxes, algo, exp, scenario, str(seed))
    #     print("evaluation is done!")

    for ft in range(11):
        print(f"Executing for fault no. {ft}.")
        for fn in range(3):
            np.random.seed(0)
            for run in range(33):
                seed = np.random.randint(0, 999)
                run_evaluation(env, num_agents, num_boxes, algo, exp, scenario, str(seed), 
                            ft, fn)
        print(f"Done executing for fault no. {ft}.")
            


if __name__ == "__main__":
    main()