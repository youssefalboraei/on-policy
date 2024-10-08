#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

from onpolicy.config import get_config
from onpolicy.envs.swarm.swarm_env import SwarmEnvWrapper
from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

from onpolicy.runner.shared.swarm_runner import SwarmRunner

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SwarmEnv":
                env = SwarmEnvWrapper(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
        
def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = SwarmEnvWrapper(all_args)
            env.seed(all_args.seed * 50000 + rank * 100000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='single_transport', help="Which scenario to run on")
    parser.add_argument("--num_faults", type=int, default=0, 
                        help="number of faults")
    parser.add_argument('--fault_type', type=int,
                        default=0, help="type of fault")
    parser.add_argument('--num_agents', type=int,
                        default=10, help="number of robots")
    parser.add_argument('--num_boxes', type=int,
                        default=10, help="number of boxes")
    parser.add_argument('--num_mboxes', type=int,
                        default=0, help="number of mboxes")
    parser.add_argument('--fault_number', type=int,
                        default=0, help="number of faulty agents")
    parser.add_argument('--delivery_bias', type=int,
                        default=1, help="delivery bias")
    parser.add_argument('--arena_width', type=int,
                        default=500, help="arena width")
    parser.add_argument('--arena_height', type=int,
                        default=500, help="arena height")
    # parser.add_argument('--model_dir', type=str,
    #                     default=None, help="model directory")

                        

    all_args = parser.parse_known_args(args)[0]

    return all_args
    # Add any other SwarmEnv-specific arguments here

    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert all_args.use_eval, ("u need to set use_eval to be True")
    print(all_args.model_dir)
    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="evaluation",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args)
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # print(all_args)
    # exit()

    # run experiments
    if all_args.share_policy:
    
        from onpolicy.runner.shared.swarm_runner import SwarmRunner as Runner
    else:
        from onpolicy.runner.separated.swarm_runner import SwarmRunner as Runner

    runner = SwarmRunner(config)
    runner.eval(total_num_steps=10_000)
    
    # post process
    eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()

if __name__ == "__main__":
    main(sys.argv[1:])