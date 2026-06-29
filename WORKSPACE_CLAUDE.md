# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout

This is a workspace, not a single project. It bundles three sibling git repos:

- `on-policy/` — actively maintained fork of the MAPPO benchmark (Yu et al., 2022) extended with a custom `SwarmEnv`. **It has its own `on-policy/CLAUDE.md` with detailed MAPPO architecture notes — read it before touching anything under `on-policy/`.**
- `cpp-simulator/` — **canonical** C++ warehouse swarm simulator + pybind11 wrapper (`warehouse_sim_cpp/scripts/marl_sim.cpp`). Produces `marl_sim.so` that `SwarmEnv.py` imports. Use this for any rebuild; the `Simulator{2,3, 4}/` trees under `msc_dissertation/` are older snapshots that lack `BY_MARL_SINGLE`.
- `msc_dissertation/` — MSc dissertation material for Youssef Alboraei (University of Bristol, ematm0055-2023). Contains older C++ simulator snapshots, Flask visualisation app, PDFs (`ants_2962_lee.pdf` is the upstream Lee & Hauert ANTS 2022 paper; `RL_Fault_Proposal_Draft.pdf` is the MSc proposal), wandb CSV exports, and `plot.py` (paper figure generator).

Plus a `training_logs/` directory at the workspace root that holds stdout from background training runs.

The top-level `msc_codebase/` directory is not itself a git repo. (Note: directory is currently named `mcs_codebase` on disk; rename to `msc_codebase` to match the dissertation acronym — must be done outside this session since Claude is `cwd`'d inside it.)

## The cross-tree dependency that makes this a single workspace

`on-policy/onpolicy/envs/swarm/SwarmEnv.py` does `import marl_sim`. The canonical source for that module is `cpp-simulator/warehouse_sim_cpp/scripts/marl_sim.cpp` — a pybind11 wrapper that exposes `Config`, `FaultManagementSimulator`, the `M_type.BY_MARL_SINGLE` enum, the `_blackboard_` struct (with the per-robot/per-box state fields SwarmEnv reads), and a `step()` path that consumes `bb.r_mitigation_action[i]` per agent.

The `Simulator2/warehouse_sim_cpp/src/wrapper/simulator_wrapper.cpp` under `msc_dissertation/` is an **older minimal wrapper** that exposes only a `SimulatorWrapper` class — insufficient for SwarmEnv. Don't build from there.

A working SwarmEnv training run needs **both** trees: `cpp-simulator/` built into an importable `marl_sim.so`, plus the on-policy MAPPO trainer.

### Build & install marl_sim (verified Linux + Python 3.12 + Blackwell GPU)

```bash
# C++ deps (header-only, lives at ~/.local/cxx-deps/)
mkdir -p ~/.local/cxx-deps/include
curl -sSfL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp \
     -o ~/.local/cxx-deps/include/nlohmann/json.hpp
git clone --depth 1 --branch 1.0.1 https://github.com/g-truc/glm /tmp/glm-src
cp -r /tmp/glm-src/glm ~/.local/cxx-deps/include/
# Stub nlohmann_jsonConfig.cmake so find_package(nlohmann_json) succeeds:
mkdir -p ~/.local/cxx-deps/share/cmake/nlohmann_json
cat > ~/.local/cxx-deps/share/cmake/nlohmann_json/nlohmann_jsonConfig.cmake <<'EOF'
if(NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../../include")
endif()
set(nlohmann_json_FOUND TRUE)
EOF
cat > ~/.local/cxx-deps/share/cmake/nlohmann_json/nlohmann_jsonConfigVersion.cmake <<'EOF'
set(PACKAGE_VERSION "3.11.3")
set(PACKAGE_VERSION_COMPATIBLE TRUE)
set(PACKAGE_VERSION_EXACT FALSE)
EOF

# Python deps (in a venv — Python 3.12 here)
python3 -m venv ~/.venv/marl
~/.venv/marl/bin/pip install --upgrade pip cmake 'pybind11<3' numpy
# pybind11 3.x rejects the bMetrics binding form; stick to 2.x.

# Build the wrapper
cd cpp-simulator/warehouse_sim_cpp
mkdir -p build && cd build
~/.venv/marl/bin/cmake .. \
  -DCMAKE_PREFIX_PATH="$HOME/.local/cxx-deps;$HOME/.venv/marl/lib/python3.12/site-packages/pybind11/share/cmake/pybind11" \
  -DCMAKE_CXX_FLAGS="-I$HOME/.local/cxx-deps/include -O2 -fPIC -DGLM_ENABLE_EXPERIMENTAL"
~/.venv/marl/bin/cmake --build . --target marl_sim -j$(nproc)

# Install
cp marl_sim.so ~/.venv/marl/lib/python3.12/site-packages/marl_sim.so
```

GLM 1.0.1 requires `-DGLM_ENABLE_EXPERIMENTAL`; older GLM didn't.

### Build & install marl_sim — required code fixes vs upstream cpp-simulator

`cpp-simulator/` as it lives on origin doesn't build cleanly with modern Python/pybind11. The fixes are committed locally on workspace `cpp-simulator/` (`git log --oneline` shows them as the most recent commit); push to origin if you want them shared:

- `CMakeLists.txt`: drop `EXACT` from `find_package(Python3 3.6 EXACT REQUIRED ...)` so non-3.6 Python builds work.
- `scripts/marl_sim.cpp`: `bMetrics::getMetric` is static — `.def("getMetric", ...)` → `.def_static(...)` (pybind11 ≥2.10 asserts).
- `scripts/marl_sim.cpp`: add `def_readonly` for 9 `_blackboard_` fields that SwarmEnv reads but the original bindings missed: `r_nearest_box_id, r_nearest_robot_id, r_nearest_wall_id, r_bid, r_messages_r, r_messages_s, r_delivered, r_pos_x, r_pos_y`. Without these, SwarmEnv crashes with `AttributeError` on the first observation step.

## on-policy branches

After the recent history rewrite, `main` is the active baseline that previously lived only on `no-global` / `global-10` (real action plumbing, 15-action space, `marl_sim` import, reward `DELIVERY=500`). The older branches are now mostly stale snapshots; `no-global-ubuntu` is the newest experimental branch.

| Branch | Status | Notes |
| --- | --- | --- |
| `main` | **Active baseline.** | Imports `marl_sim`. Action dim **15**. Action plumbed (`mitigation_actions[i] = action[0]`). Reward `DELIVERY=500, DISTANCE=0.01, DROP=0.5, TIME=0.01`. Per-agent obs includes `r_messages_r/s`, `r_delivered`. `train_swarm_scripts/` only has `train_swarm_single.{py,sh}` and `train_swarm_collective.py` (no `_small`/`_global_10` variants). |
| `no-global-ubuntu` | **Newest experimental branch.** | Adds `onpolicy/envs/swarm/test_env.py` (standalone SwarmEnv smoke test, 3 agents / 250×250 / fault N2T3). Modifies `algorithms/mat/algorithm/transformer_policy.py` and both shared/separated `base_runner.py`. Adds `onpolicy/utils/{action_tracker,data_collector,visualise}.py`. Inherits the no-shared-obs variant of SwarmEnv. |
| `no-global` | Stale snapshot. | Older variant where `share_observation_space` did not concat global state. Adds `onpolicy/utils/{action_tracker,data_collector,visualise}.py`. Explored fault types {5, 8, 9} (only branch with ft=9). |
| `global-10` | Stale snapshot. | 10-agent 500×500 variant of the small-env experiments. Adds `train_swarm_scripts/train_swarm_single_global_10.py` (scenario `single_transport_small` but with `num_agents=10`, dynamic fault sampling). |
| `small-env` | Stale snapshot. | 3-agent 250×250 small-environment training variant. `steps_per_iteration=150`, `episode_limit=500`. Adds `train_swarm_scripts/train_swarm_single_small.py`. |
| `small-env-eval` | **Source of all evaluation data.** | Forked off `small-env` for evaluation sweeps. Disables agent action (`mitigation_actions[i] = 0`) for the baseline condition. Holds the entire RT/ST × RE/SE comparison dataset, the U-test analysis, and every box plot / action-frequency script (see "Evaluation framework" below). |

If `git status` shows `Your branch is ahead/behind`, history was rewritten on origin. The pre-rewrite local HEAD was tagged `backup-main-pre-rewrite-<date>` before the reset that aligned local main to origin/main.

## Evaluation framework — `small-env-eval` (RT/ST × RE/SE)

Everything in `on-policy/onpolicy/tests/` is from the `small-env-eval` branch. The naming scheme:

- **RE = Reduced Environment** — 3 robots × 3 boxes in a 250×250 cm arena (scenario `single_transport_small`).
- **SE = Standard Environment** — 10 robots × 10 boxes in a 500×500 cm arena (scenario `single_transport` / `single_transport_share_obs`).
- **RT** and **ST** — two trained MARL model variants being compared against the same `without_mitigation` (no-MARL) baseline. Both labels are confirmed by `tests/box_plot_combined/combined-scenarios-boxplot-script.py` which pairs `RT-RE`+`ST-RE` for the Reduced env and `RT-SE`+`ST-SE` for the Standard env on the same boxplot. The script comments suggest these correspond to `rmappo` (recurrent) vs `mappo` (stacked-frames / non-recurrent) checkpoints, but the model→label mapping isn't named explicitly anywhere in code — confirm by checking `--model_dir` used to generate each bucket if it matters.

Per-bucket directory layout:

```
onpolicy/tests/<RT|ST>-<RE|SE>/
  with_mitigation/                  # MARL agent enabled (action[0] written to bb.r_mitigation_action)
    simulation_data_run<N>_N<f>T<t>.csv
  without_mitigation/               # Baseline: mitigation_actions[i]=0; base swarm is ~a random walk
    simulation_data_run<N>_N<f>T<t>.csv
  performance_comparison_results.csv  # one row per run: Fault_Number, Fault_Type, Fault_Name, P_Mitigated, P_Baseline, P_Difference
  summary_statistics.csv              # aggregated by Fault_Name: mean/median/std of each P_* (only present in RT-SE and ST-RE)
  performance_comparison_boxplot.png
```

Per-step CSV schema (in `with_mitigation/` and `without_mitigation/`):
```
Run, Fault_Number, Fault_Type, Seed, Step, Delivery_Rate, Robot_0_Action, ..., Robot_<N-1>_Action
```
Number of `Robot_*_Action` columns matches the env scale: 3 for RE, 10 for SE.

Performance metric (from `combined-scenarios-boxplot-script.py:calculate_performance`):
- SE: `df['Delivery_Rate'].sum() / 10 / 50`
- RE: `df['Delivery_Rate'].sum() / 3 / 20` (commented variant)

Labelled **N-CBDS** ("normalised cumulative box-delivery score" — inferred from the y-axis label) in plots.

**Eval-harness gotcha:** N-CBDS sums `Delivery_Rate` over a *fixed* window (50 SE / 20 RE steps). An eval driver must run that full fixed step count and keep recording after task completion (`Delivery_Rate` then sits at its max). An eval that breaks the episode on `all(dones)` writes a short CSV, and N-CBDS silently under-scores fast policies because the post-completion max-value steps are missing. The historical RT/ST eval ran the fixed window; `eval_nofault.py` was fixed on 2026-05-21 to match, after break-on-done truncation made the no-fault models look worse than the random-walk baseline.

### Fault grid

CSV filenames encode the (faults, type) cell as `N<f>T<t>`. From `combined-scenarios-boxplot-script.py:FAULT_NAMES`:

| `T` | Name | In dataset? |
| --- | --- | --- |
| 0 | NO_FAULT / DEFAULT | yes (N0T0 only) |
| 1 | SINGLE_WHEEL | no |
| 2 | DOUBLE_WHEEL | no |
| **3** | **F1: ALL_WHEEL_V0** | yes |
| **4** | **F2: ALL_WHEEL_V10** | yes |
| **5** | **F3: ALL_WHEEL_V50** | yes |
| 6 | RADIAL_CAM_4 | no |
| 7 | UPFACING_CAM | no |
| **8** | **F4: PICKUP** | yes |
| 9 | DROPOFF | no |
| 10 | LASER_16 | no |
| 11 | R_COMMS | no |
| 12 | S_COMMS | no |

Only 5 fault names appear in `summary_statistics.csv` (DEFAULT plus F1-F4). The wider `T` range exists in the source code but the actual evaluation matrix is `T ∈ {0, 3, 4, 5, 8}` × `N ∈ {0, 1, 2, 3}` for RE (4 robots max) and a wider `N ∈ {0..10}` for SE.

Files per bucket: RT-RE 2 600, RT-SE 8 208, ST-RE 2 613, ST-SE 8 210.

### Analysis scripts (`onpolicy/tests/`)

| Path | Purpose |
| --- | --- |
| `box_plot/box_plot.py` | Per-bucket boxplot — `Performance` vs `Fault_Name`, hue=`Condition` (Baseline / Mitigated). |
| `box_plot_combined/combined-scenarios-boxplot-script.py` | Argv: `RE` or `SE`. Pairs RT+ST against shared baseline for one env scale. Writes `rt_<env>_st_<env>_performance_comparison_boxplot.png` and a paired `summary_statistics.csv`. |
| `box_plot_combined/all-scenarios-boxplot-script.py` | Variant that combines all four buckets. |
| `u-test/u-test.py`, `u-test-updated.py` | Mann-Whitney U test per (fault_number, fault_type) cell, comparing `with_mitigation` vs `without_mitigation`. Outputs `Mitigation_Power = 2 * sign(R_mit - R_none) * |U/(n1*n2) - 0.5|` ∈ [-1, +1]. Generates `rt_re_*.csv/png`, `rt_se_*.csv/png`, `st_re_*.csv/png`, `st_se_*.csv/png` heatmaps + the combined `performance_analysis.csv` and `performance_comparison.png`. |
| `action_frequency/`, `action_frequency_signature{,_change,_n}/`, `action_frequency_analysis/` | Action histograms / signatures per fault / robot. The 13 action labels live at the top of `data_collector.py`: `NO_ACTION, DECREASE_SPEED_50, STOP_MOVING, BIAS_TO_NEAREST_{ROBOT,BOX,WALL}, BIAS_LEFT, BIAS_FROM_NEAREST_{ROBOT,BOX,WALL}, ATTRACT_NEIGHBOUR, REPEL_NEIGHBOUR, DROP_BOX`. |
| `best/`, `best_2/`, `best_a99/`, `best_cih/` | Checkpointed `actor.pt`/`critic.pt` for the best models per scenario. `best_cih/` also keeps the wandb export CSV that selected it. |
| `data/`, `data_2/`, `data_backup/` | Older intermediate eval data — superseded by `RT-RE`, `RT-SE`, `ST-RE`, `ST-SE`. `data_backup/` mirrors `box_plot/` and `u-test/` script copies. |
| `rt-curve.png`, `st-curve.png` | Training-curve images for the two model variants. |

### Re-generating the evaluation data

`eval_swarm_mappo_ubuntu_runs.py` is the Linux-aware driver: it loops `for ft in range(11): for fn in range(3): for run in range(33): seed = np.random.randint(0,999)` and shells out to `eval/eval_swarm.py` per cell. Tweak the `if ft < 6: continue` guard to re-run a subset. The Windows-only siblings (`eval_swarm_mappo.py`, `eval_swarm_mappo_ubuntu.py`) hardcode `D:\youssef\on-policy\...` paths.

## Simulator versions

The canonical simulator is `cpp-simulator/` at the workspace root (see "Build & install marl_sim" above). `msc_dissertation/` also has older C++ snapshots:

| Path | Status |
| --- | --- |
| `cpp-simulator/warehouse_sim_cpp/` | **Canonical.** Source for `marl_sim.so`. Use for any rebuild. |
| `msc_dissertation/Simulator2/warehouse_sim_cpp/` | Older snapshot. Has a minimal `src/wrapper/simulator_wrapper.cpp` that exposes only `SimulatorWrapper` (not `Config`/`FaultManagementSimulator`/`M_type.BY_MARL_SINGLE`) — insufficient for SwarmEnv. |
| `msc_dissertation/Simulator3/warehouse_sim_cpp/` | Older iteration; diverged `blackboard.h`, `FaultManagementSimulator.h`, `bMetrics.cpp`. No wrapper. Lacks the MARL enum entirely. |
| `msc_dissertation/Simulator 4/warehouse_sim_cpp/` | Latest msc_dissertation iteration. Adds `obj/Agent.{h,cpp}` and `obj/Object.{h,cpp}`, refactors many headers. `M_type` only has `BY_MODEL`, no MARL path. No wrapper. |
| `msc_dissertation/Simulator/`, `fkvizsim/`, `fkvizsim-2/`, `MARL/{codebase,MARLlib,on-policy}/`, `MARL/MADDPG/Multi-Agent-Reinforcement-Learning/`, `msc_dissertation/on-policy/`, `ematm0055-2023-youssefalboraei/` | Empty placeholders. Do not invent content for them. |

`msc_dissertation/envs/` is a conda environment (with `bin/`, `lib/`, `share/`, …) that was built on a different OS (macOS likely) — the binaries fail with "Exec format error" on Linux. Ignore when reading code, do not try to use the Python interpreter inside.

## Building the C++ simulator

See "Build & install marl_sim" above for the canonical `cpp-simulator/` build. For the older `msc_dissertation/Simulator2/` (only useful for the standalone executables — `viz_sim`, `metrics_sim`, `mitgn_sim`, `neurov_sim`, `test*` — not for `marl_sim`):

```bash
cd msc_dissertation/Simulator2/warehouse_sim_cpp
mkdir -p build && cd build
cmake ..
cmake --build .
```

Dependencies: `nlohmann_json >= 3.7.0`, GLM, pybind11.

Short-key arg vocabulary (shared by all C++ binaries and `SwarmEnv`'s `config_args` dict):

| Flag | Meaning |
| --- | --- |
| `-ft`, `-fn` | fault type / number of faults |
| `-aw`, `-ah` | arena width / height (default 500×500 cm) |
| `-a`, `-b`, `-bm` | number of agents / boxes / "m" (collective-transport) boxes |
| `-i`, `-s` | iterations / random seed |
| `-srid`, `-cwait`, `-ctime` | sample robot ids, team-leave wait, rejoin buffer |
| `-mid`, `-mb`, `-dbias` | hardcoded mitigation id / buffer before mitigation / delivery-zone bias |
| `--compute-delr`, `--compute-metr`, `--sample-metr` | metrics toggles |
| `--exit-on-compl`, `--pred-fault` | exit on completion / enable fault prediction |

The `install(TARGETS ... DESTINATION /home/bk21562/git/fkvizsim/src)` rule in every `CMakeLists.txt` is the original author's local path — `make install` will fail outside that machine. Skip install or rewrite the destination.

## Flask viz app

`msc_dissertation/Simulator{2,3, 4}/warehouse_viz/app.py` is a small Flask app (port 5000, debug mode) serving HTML/JS visualisation. It shells out to pre-built executables (`viz_sim`, `trial`, `rates`) that must exist in `src/`. The three ELF binaries checked into `Simulator2/warehouse_viz/src/` are an existing build; on a clean checkout you must build them yourself from the C++ tree.

`G_DATA_FILES` in `src/config.py` enumerates the per-run output files the C++ binaries drop into `src/` and the app moves into `data/<controller>/`: `robots.txt, boxes.txt, boxes_m.txt, dtags.txt, heading.txt, c_heading.txt, metadata.txt, metrics.txt`.

## Tests

No central test runner. Standalone smoke tests:

- `msc_dissertation/test_wandb.py` — `wandb.init` round-trip with CIFAR-100 placeholders (not a real ML test).
- `on-policy/onpolicy/tests/test_swarm_mappo{,2}.py` — manual SwarmEnv-with-MAPPO smoke runs (load `actor.pt`/`critic.pt` from the same dir).
- `on-policy/onpolicy/envs/swarm/test_env.py` (on `no-global-ubuntu` only) — standalone SwarmEnv constructor + 10-step smoke loop with 3 agents / 250×250 / N2T3.
- C++ side: `Simulator{2,3, 4}/warehouse_sim_cpp/scripts/tests/nn_test*.cpp` build as standalone executables via the per-simulator `CMakeLists.txt`.

## Paper figures & data

- `msc_dissertation/plot.py` reads a fixed-name wandb export CSV (`wandb_export_2025-10-05T01_53_47.941+01_00.csv`) and emits `figure2_camera_ready.{pdf,png,eps}`. Smoothing window 25, single column `rmappo_global-10_seed1 - average_episode_rewards`. Edit the filename and column at the top to reuse.
- Two `wandb_export_*.csv` snapshots at the dissertation root (2024-08-26 / 2025-10-05).
- `msc_dissertation/wandb/` is a 2024-07-25 run dir from a previous session (test_wandb smoke) — historical, safe to ignore.
- `ieeeconf.zip` is just the IEEE LaTeX template — no actual paper content.

## Conventions worth knowing

- The `marl_sim` import name was `marl_sim_2` on old `main` only — the active codebase everywhere else uses unsuffixed `marl_sim`. If both aliases were ever installed, switching machines may pull a different `.so` than expected. Match the build name to what `SwarmEnv.py` imports on the branch you're on.
- Several `Simulator*` snapshots are deliberate progress-checkpoints with diverging core headers — they are **not** interchangeable. The pybind11 wrapper exists only in `Simulator2`.
- All `eval_swarm_mappo*.py` scripts except `eval_swarm_mappo_ubuntu_runs.py` (and `.._ubuntu.py`) hardcode Windows paths `D:\youssef\on-policy\...` and UNC `\\nstu-nas01.uwe.ac.uk\...` — rewrite before running on Linux.
- `.gitignore` on new main ignores `*.csv` and a `RODE/` directory; the `RODE` entry is aspirational (no such dir exists in any branch yet).
- The `with_mitigaion/` typo (single 'g') appears in checked-in dir paths under `tests/data/with_mitigaion/`. Keep the typo when matching code to data; the newer `tests/{RT,ST}-{RE,SE}/with_mitigation/` uses the correct spelling.
- `.DS_Store` files and the `suet_lee-warehouse_*.zip` archives at `msc_dissertation/` root are upstream/dissertation provenance — not active source.
