#!/usr/bin/env python
"""Run the a99 (ST) rmappo policy live and dump the PER-TICK swarm trajectory.

Mirrors onpolicy/scripts/eval/eval_swarm.py for policy construction / restore.
The C++ sim runs steps_per_iteration=200 internal ticks per RL step and write_viz
logs one line per tick (50 ticks/sec, dt=0.02s), so we read those per-tick files
back as the true physics trajectory and hold each RL step's action across its
ticks. Render is a separate step (render_a99.py).
"""
import os
import sys
import json
import types
import argparse
import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.envs.swarm.swarm_env import SwarmEnvWrapper
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

HERE = os.path.dirname(os.path.abspath(__file__))
A99_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "tests", "best_a99"))


def build_args(extra):
    """Reproduce the eval CLI (eval_swarm_mappo_ubuntu_runs.py) but at SE scale."""
    parser = get_config()
    # SwarmEnv-specific args (same names eval_swarm.py adds)
    parser.add_argument('--scenario_name', type=str, default='single_transport')
    parser.add_argument('--num_faults', type=int, default=0)
    parser.add_argument('--fault_type', type=int, default=0)
    parser.add_argument('--num_agents', type=int, default=10)
    parser.add_argument('--num_boxes', type=int, default=10)
    parser.add_argument('--num_mboxes', type=int, default=0)
    parser.add_argument('--fault_number', type=int, default=0)
    parser.add_argument('--delivery_bias', type=int, default=1)
    parser.add_argument('--arena_width', type=int, default=500)
    parser.add_argument('--arena_height', type=int, default=500)
    all_args = parser.parse_known_args(extra)[0]
    return all_args


def main(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--fault_type', type=int, default=3)
    ap.add_argument('--num_faults', type=int, default=3)
    ap.add_argument('--seed', type=int, default=708)
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--model_dir', type=str, default=A99_DIR)
    ap.add_argument('--baseline', action='store_true',
                    help="no mitigation: force all actions to NO_ACTION (random-walk base)")
    ap.add_argument('--fixed_ticks', type=int, default=0,
                    help="run exactly this many internal ticks (ignore completion); "
                         "use to match a mitigation run's duration for a fair baseline")
    known, _ = ap.parse_known_args(argv)

    # Exactly the eval driver's policy hyperparams (ST / a99 = rmappo, stacked frames)
    eval_cli = [
        "--env_name", "SwarmEnv",
        "--cuda", "False",
        "--algorithm_name", "rmappo",
        "--experiment_name", "render",
        "--scenario_name", "single_transport",
        "--num_agents", "10",
        "--num_boxes", "10",
        "--seed", str(known.seed),
        "--arena_height", "500",
        "--arena_width", "500",
        "--delivery_bias", "1",
        "--n_rollout_threads", "1",
        "--n_eval_rollout_threads", "1",
        "--num_mini_batch", "1",
        "--episode_length", str(known.steps),
        "--ppo_epoch", "10",
        "--use_ReLU",
        "--gain", "0.01",
        "--lr", "7e-4",
        "--critic_lr", "7e-4",
        "--use_eval",
        "--stacked_frames", "4",
        "--use_stacked_frames",
        "--hidden_size", "128",
        "--layer_N", "3",
        "--model_dir", known.model_dir,
        "--num_faults", str(known.num_faults),
        "--fault_type", str(known.fault_type),
    ]
    all_args = build_args(eval_cli)
    # rmappo => recurrent (eval_swarm.py)
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False

    device = torch.device("cpu")
    torch.set_num_threads(1)
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    env = SwarmEnvWrapper(all_args)
    env.seed(all_args.seed)

    # SwarmEnv._is_done has a hardcoded debug `exit()` at step>200 that kills the
    # process. Override it (driver-local, file untouched) so the episode runs to
    # natural completion (all boxes delivered) with our own safety cap instead.
    def _is_done_safe(self):
        delivered = self.simulator.bb.s_delivery_rate[-1] >= self.num_boxes
        capped = self.step_counter >= known.steps
        done = bool(delivered or capped)
        return {a: done for a in self.agents}

    env.env._is_done = types.MethodType(_is_done_safe, env.env)

    policy = None
    if not known.baseline:
        share_obs_space = env.share_observation_space if all_args.use_centralized_V else env.observation_space[0]
        policy = Policy(all_args, env.observation_space[0], share_obs_space,
                        env.action_space[0], device=device)
        actor_sd = torch.load(os.path.join(known.model_dir, "actor.pt"), map_location=device)
        policy.actor.load_state_dict(actor_sd)
        policy.actor.eval()

    num_agents = all_args.num_agents
    recurrent_N = all_args.recurrent_N
    hidden = all_args.hidden_size

    # write_viz dumps one line per INTERNAL TICK (50 ticks/sec, dt=0.02s) to the
    # cwd. steps_per_iteration=200 ticks per RL step, so the per-tick viz files are
    # the true physics trajectory. Run in an isolated dir and read them back; the
    # per-RL-step action is held constant across that step's ticks.
    vizdir = os.path.abspath(known.out) + ".vizdir"
    os.makedirs(vizdir, exist_ok=True)
    for f in ("robots.txt", "boxes.txt", "heading.txt", "metadata.txt"):
        p = os.path.join(vizdir, f)
        if os.path.exists(p):
            os.remove(p)
    prev_cwd = os.getcwd()
    os.chdir(vizdir)

    arena_w = env.env.config.arena_width
    arena_h = env.env.config.arena_height
    faulted = list(range(known.num_faults))

    def n_ticks():
        try:
            with open("robots.txt", "rb") as fh:
                return sum(1 for _ in fh)
        except FileNotFoundError:
            return 0

    obs, _ = env.reset()                       # truncates viz files, ticks=0
    rnn_states = np.zeros((num_agents, recurrent_N, hidden), dtype=np.float32)
    masks = np.ones((num_agents, 1), dtype=np.float32)
    bb = env.env.simulator.bb

    step_actions = []      # (n_steps, num_agents) action taken that RL step
    step_bounds = []       # cumulative tick count after each RL step
    step_drate = []        # delivery rate after each RL step
    zero_action = np.zeros((num_agents, 1), dtype=np.int64)   # NO_ACTION baseline
    for step in range(known.steps):
        if known.baseline:
            action = zero_action                          # no mitigation
        else:
            with torch.no_grad():
                act_t, rnn_states = policy.act(
                    np.array(obs), rnn_states, masks, deterministic=True)
            action = act_t.detach().cpu().numpy()         # (num_agents, 1)
            rnn_states = rnn_states.detach().cpu().numpy()

        step_actions.append(action[:, 0].astype(np.int32))
        obs, _, rew, done, info = env.step(action)
        step_bounds.append(n_ticks())
        step_drate.append(float(bb.s_delivery_rate[-1]))
        if known.fixed_ticks > 0:
            if n_ticks() >= known.fixed_ticks:            # match a fixed duration
                break
        elif np.all(done):
            break
    env.close()

    # read per-tick trajectory. Once all boxes are delivered the C++ writer emits
    # a malformed `}` for boxes.txt; carry forward the last valid box map so the
    # (now-deposited) boxes stay shown in place.
    def read_jsonl(fname):
        out = []
        for l in open(fname):
            s = l.strip()
            try:
                out.append(json.loads(s))
            except Exception:
                out.append(None)          # malformed `}` after all boxes delivered
        return out

    robots = read_jsonl("robots.txt")
    boxes = read_jsonl("boxes.txt")
    headings = read_jsonl("heading.txt")
    os.chdir(prev_cwd)

    T = len(robots)
    if known.fixed_ticks > 0:
        T = min(T, known.fixed_ticks)        # exact duration match for fair baseline
    rkeys = sorted(robots[0].keys(), key=int)
    # boxes get removed from the map once delivered (and the tail goes malformed);
    # use the full initial key set and mark absent boxes NaN so they vanish.
    bkeys = sorted(boxes[0].keys(), key=int)
    rx = np.array([[robots[t][k][0] for k in rkeys] for t in range(T)], dtype=np.float32)
    ry = np.array([[robots[t][k][1] for k in rkeys] for t in range(T)], dtype=np.float32)

    def box_arr(coord):
        a = np.full((T, len(bkeys)), np.nan, dtype=np.float32)
        for t in range(T):
            bt = boxes[t]
            if isinstance(bt, dict):
                for j, k in enumerate(bkeys):
                    if k in bt:
                        a[t, j] = bt[k][coord]
        return a

    bx, by = box_arr(0), box_arr(1)
    head = np.array([headings[t] if headings[t] is not None else headings[t - 1]
                     for t in range(T)], dtype=np.float32)

    # expand per-RL-step actions / drate to per-tick using the tick boundaries
    acts = np.zeros((T, num_agents), dtype=np.int32)
    drate = np.zeros(T, dtype=np.float32)
    prev = 0
    for s, b in enumerate(step_bounds):
        b = min(b, T)
        acts[prev:b] = step_actions[s]
        drate[prev:b] = step_drate[s]
        prev = b
    if prev < T:                               # any trailing ticks
        acts[prev:] = step_actions[-1]
        drate[prev:] = step_drate[-1]

    out = dict(rx=rx, ry=ry, bx=bx, by=by, head=head, actions=acts, drate=drate)
    out["faulted"] = np.array(faulted, dtype=np.int32)
    out["baseline"] = np.int32(1 if known.baseline else 0)
    out["ticks_per_sec"] = np.int32(50)        # postIterate dt = 0.02s
    out["arena_w"] = np.float32(arena_w)
    out["arena_h"] = np.float32(arena_h)
    out["fault_type"] = np.int32(known.fault_type)
    out["num_faults"] = np.int32(known.num_faults)
    out["seed"] = np.int32(known.seed)
    np.savez_compressed(known.out, **out)
    print(f"[capture] {len(step_actions)} RL steps -> {T} ticks "
          f"({T/50:.1f}s sim) -> {known.out}  "
          f"final delivery_rate={out['drate'][-1]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
