# Handoff: continuing the swarm-RL evaluation work on a new machine

Last updated: 2026-06-29. This file lives in the `on-policy` repo (`small-env-eval`
branch) so it travels with `git clone`. The workspace-root `CLAUDE.md` does **not**
clone (it sits above any repo) — copy it separately if you want the full project notes.

## TL;DR

The recent work (≤3-faulty-robot SE figures for STNF-SE / ST-SE, plus a corrected
full mitigation-power heatmap) is committed and pushed to
`git@github.com:youssefalboraei/on-policy.git`, branch **`small-env-eval`**
(commits `bbc8ea4d9` and `52a30418f`). On the new machine you mainly need to:
clone three repos, **re-apply the uncommitted `cpp-simulator` build fixes**, build
`marl_sim.so`, make a venv, and you can re-run the figure scripts.

## 0. Moving the whole workspace (recommended: full transfer)

To be sure nothing is missed (uncommitted changes, non-git files, data), copy the
entire workspace, then rebuild the venv. The venv (`~/.venv/marl`, 7.4G) and
`marl_sim.so` live OUTSIDE the workspace and are machine-specific binaries — do NOT
copy them, rebuild per section 2.

Workspace is 5.8G; skip the dead weight (the 1.5G broken macOS conda env, caches,
build dirs) to move ~4G:

```bash
# direct machine-to-machine (preferred if you have ssh between them)
rsync -avzP \
  --exclude 'msc_dissertation/envs/' \
  --exclude '__pycache__/' \
  --exclude '*/build/' \
  --exclude '.venv/' \
  ~/projects/mcs_codebase/  USER@NEWHOST:~/projects/mcs_codebase/

# OR a portable archive (USB / cloud); add --exclude='conference_videos' to drop 1.2G
tar czf ~/mcs_codebase_transfer.tar.gz \
  --exclude='msc_dissertation/envs' --exclude='__pycache__' --exclude='*/build' \
  -C ~/projects mcs_codebase
```

After copying, fix the git worktree link (`on-policy-eval/` is a linked worktree of
`on-policy/`; its `.git` holds an absolute path):

```bash
cd ~/projects/mcs_codebase/on-policy && git worktree repair
```

Then do section 2 (build `marl_sim.so` + venv) and you can continue working.

Lighter alternative (git-clone route): both active repos are fully pushed
(`on-policy@small-env-eval`, `cpp-simulator@win11`), so you can instead `git clone`
them (section 1) and rsync only the non-git extras (section 6) — smaller, and no
worktree repair needed, but you must remember the extras.

## 1. Workspace layout (three separate git repos)

Put all three side by side in one workspace dir (e.g. `~/projects/mcs_codebase/`):

| Dir | Repo | Branch to use | Role |
| --- | --- | --- | --- |
| `on-policy/` | `git@github.com:youssefalboraei/on-policy.git` | `small-env-eval` | MAPPO trainer + SwarmEnv + **all eval data & figures** |
| `cpp-simulator/` | `git@github.com:youssefalboraei/cpp-simulator.git` | `win11` | C++ sim + pybind11 wrapper -> `marl_sim.so` |
| `msc_dissertation/` | `git@github.com:youssefalboraei/msc_dissertation.git` | `main` | dissertation material, older sim snapshots, PDFs |

```bash
mkdir -p ~/projects/mcs_codebase && cd ~/projects/mcs_codebase
git clone -b small-env-eval git@github.com:youssefalboraei/on-policy.git
git clone -b win11          git@github.com:youssefalboraei/cpp-simulator.git
git clone -b main           git@github.com:youssefalboraei/msc_dissertation.git
```

`SwarmEnv.py` does `import marl_sim` — the module's only source is
`cpp-simulator/warehouse_sim_cpp/scripts/marl_sim.cpp`. A working run needs BOTH the
`cpp-simulator` build and the `on-policy` trainer.

## 2. Build `marl_sim.so` (verified: Linux + Python 3.12)

### 2a. `cpp-simulator` build fixes — now committed

The Linux/modern-pybind11 build fixes are committed and pushed to `origin/win11`
(commit `d4a0d6e`), so a fresh `git clone -b win11` builds cleanly — no manual edits
needed. For reference they: guard the Windows-only CONDA_PREFIX block behind WIN32;
drop `EXACT` from `find_package(Python3 3.6 ...)`; treat GLM as header-only on Linux;
`#include_next` the system POSIX `<dirent.h>` on non-Windows; and add the
`Config.write_viz` binding. (Older note: the `install(... DESTINATION /home/bk21562/...)`
rule is the original author's path — skip `make install` or rewrite it.)

### 2b. Header-only C++ deps (`~/.local/cxx-deps/`)
```bash
mkdir -p ~/.local/cxx-deps/include/nlohmann
curl -sSfL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp \
     -o ~/.local/cxx-deps/include/nlohmann/json.hpp
git clone --depth 1 --branch 1.0.1 https://github.com/g-truc/glm /tmp/glm-src
cp -r /tmp/glm-src/glm ~/.local/cxx-deps/include/
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
```

### 2c. Python venv + build (pybind11 must be < 3)
```bash
python3 -m venv ~/.venv/marl
~/.venv/marl/bin/pip install --upgrade pip cmake 'pybind11<3' numpy
cd cpp-simulator/warehouse_sim_cpp && mkdir -p build && cd build
~/.venv/marl/bin/cmake .. \
  -DCMAKE_PREFIX_PATH="$HOME/.local/cxx-deps;$HOME/.venv/marl/lib/python3.12/site-packages/pybind11/share/cmake/pybind11" \
  -DCMAKE_CXX_FLAGS="-I$HOME/.local/cxx-deps/include -O2 -fPIC -DGLM_ENABLE_EXPERIMENTAL"
~/.venv/marl/bin/cmake --build . --target marl_sim -j$(nproc)
cp marl_sim.so ~/.venv/marl/lib/python3.12/site-packages/marl_sim.so
```

### 2d. Verify
```bash
~/.venv/marl/bin/python -c "import marl_sim; print('marl_sim OK')"
cd ../../../on-policy && ~/.venv/marl/bin/python -c \
  "import sys; sys.path.insert(0,'.'); from onpolicy.envs.swarm.SwarmEnv import SwarmEnv; print('SwarmEnv OK')"
```

The plotting deps (pandas/seaborn/scipy/matplotlib) also live in `~/.venv/marl`; if the
fresh venv lacks them: `~/.venv/marl/bin/pip install pandas seaborn scipy matplotlib`.

## 3. This session's work (committed in `on-policy`, branch `small-env-eval`)

All under `onpolicy/tests/`. Re-run any script with `~/.venv/marl/bin/python <script>`.

### `n_le_3/` — figures capped at ≤3 faulty robots, for STNF-SE and ST-SE
- `make_n3_figures.py` -> per bucket: performance boxplot, mitigation-power heatmap
  (+ analysis CSV), action-frequency by-fault heatmap/bar; plus a combined boxplot.
- `action_freq_signature_n3.py` -> the F/NF-subrow "signature" heatmaps (combined over
  N≤3 and one per N=0..3), matching the existing `action_frequency_signature` format.
- `boxplot_n3_vs_n10.py` -> 6-box plot: Baseline/STNF-SE/ST-SE at ≤3 vs ≤10.
- `boxplot_per_fault_n3_vs_n10.py` -> per-fault-type (DEFAULT,F1-F4), 6 boxes each
  (3 conditions x 2 ranges = 30 boxes). ColorBrewer palette (gray/blue/orange).
- Outputs: `n_le_3/<stnf-se|st-se>/*.png|csv` and the top-level comparison PNGs.

### `u-test/` — corrected full mitigation-power heatmap
- `make_stnf_se_full_heatmap.py` -> `stnf-se_mitigation_power_heatmap_full.png`
  (+ `..._analysis_full.csv`), N=0..10, rebuilt from `performance_comparison_results.csv`.
  The pre-existing `stnf-se_mitigation_power_heatmap.png` is **sign-inverted/wrong**
  (see findings) and was left in place for comparison.

## 4. Key facts / findings (so they are not re-derived)

- **Bucket naming** `[Training][NF?]-[Execution]`: R/S = Reduced/Standard *training* env
  (3 robots/250x250 vs 10/500x500); NF = trained *without* faults (faults still in the
  eval grid); RE/SE = Reduced/Standard *execution* env. So `STNF-SE` = standard-trained,
  no-fault-trained, standard-executed; `ST-SE` = standard-trained-with-faults.
- **a99 == the ST-SE bucket.** The `best_a99` checkpoint (rmappo, hidden 128, layer_N 3,
  stacked_frames 4) generated the ST-SE eval data; the conference videos are `a99_ST-SE_*`.
  There is no separate a99 eval bucket. "a99 with faults in SE" figures == ST-SE figures.
- **NF buckets have no `without_mitigation/` folder** — the no-mitigation baseline is the
  same random-walk swarm regardless of policy, so the shared baseline is already the
  `P_Baseline` column in each bucket's `performance_comparison_results.csv`.
- **`u-test/` NF heatmaps are sign-inverted** (stnf-se, rtnf-se, stnf-re, rtnf-re): they
  show the trained policy far BELOW baseline, contradicting the comparison CSV. Rebuild
  from `performance_comparison_results.csv` (`P_Mitigated` vs `P_Baseline`).
- **N-CBDS metric**: `Delivery_Rate.sum()/10/50` (SE) or `/3/20` (RE). `P_Mitigated` /
  `P_Baseline` in the comparison CSVs equal this (verified). Eval must run the FIXED
  window (50 SE / 20 RE steps) and keep recording after task completion — breaking on
  `all(dones)` writes short CSVs and under-scores fast policies.
- **`SwarmEnv._is_done` calls `exit()`** at step>200 — kills long rollouts; override in
  the driver if you script new eval runs.

## 5. Cross-repo state to clean up before relying on a clone

- `on-policy` (`small-env-eval`): clean and pushed, EXCEPT one stray untracked
  `onpolicy/metadata.txt` (a simulator run artifact — ignore or delete).
- `cpp-simulator` (`win11`): clean and pushed (build fixes committed as `d4a0d6e`).
- `msc_dissertation` (`main`): ~24 uncommitted entries, mostly an extracted
  `suet_lee-warehouse_viz-*` archive + `.pyc`/ELF build artifacts (reference material,
  not the active RL work). NOT committed — the full transfer in section 0 carries them
  as-is; the git-clone route omits them. One real source edit if you care:
  `Simulator2/warehouse_sim_cpp/src/wrapper/CMakeLists.txt`.

## 6. Non-git files to copy manually (not in any repo)

- Workspace-root `CLAUDE.md` (full project notes) — sits above all repos, won't clone.
- `training_logs/` at the workspace root (background-run stdout), if you want it.
- The Claude memory dir `~/.claude/projects/-home-...-mcs-codebase/memory/` (optional;
  Claude-specific context).

## 7. Eval data quick reference

Per bucket: `with_mitigation/simulation_data_run<N>_N<f>T<t>.csv` (columns: Run,
Fault_Number, Fault_Type, Seed, Step, Delivery_Rate, Robot_0..k_Action) and a
`performance_comparison_results.csv` (one row/run: Fault_Number, Fault_Type, Fault_Name,
P_Mitigated, P_Baseline, P_Difference). Fault grid: T in {0,3,4,5,8} =
{DEFAULT, F1 ALL_WHEEL_V0, F2 ALL_WHEEL_V10, F3 ALL_WHEEL_V50, F4 PICKUP};
N in {0..3} (RE) or {0..10} (SE).
