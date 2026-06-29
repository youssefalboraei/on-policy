# a99 ST-SE conference-video pipeline

Renders videos of the **a99** (ST, rmappo) fault-mitigation policy running in
SwarmEnv at SE scale (10 robots / 10 boxes / 500x500 cm) for the ANTS talk.

Two visual styles share one capture step:

- **fkvizsim style** — flat circles/pentagons/squares (`render_a99.py`, `split_screen.py`).
- **symbol style** — the user's hand-drawn robot/crate glyphs (`render_symbols.py`,
  `split_screen_symbols.py`). This is the current style for the talk.

## Pipeline

```
capture_a99.py  --->  *.npz (per-tick trajectory)  --->  render_*.py  --->  *.mp4
```

1. **Capture** (`capture_a99.py`) runs the policy live and dumps the **per-tick**
   trajectory to an `.npz`. It reads the C++ `write_viz` per-tick dumps
   (robots/boxes/heading) rather than sampling once per RL step.
   - Time base: 50 ticks/sec (dt=0.02s); `steps_per_iteration = 200` ticks per RL step,
     so a fresh MARL action is chosen every **4.0 s** of sim time and held constant
     (synchronous across all robots).
   - `--baseline` forces every action to NO_ACTION (0) = the random-walk baseline.
   - `--fixed_ticks N` runs exactly N ticks (match a mitigation run for a fair clock).
   - Works around the hardcoded `exit()` in `SwarmEnv._is_done` (monkeypatched).

2. **Render** any `.npz` to MP4 as a timelapse with an honest real-time clock
   (`--speed` x real time, `--trail_seconds` fade in real sim-seconds).

## Scripts

| Script | What |
| --- | --- |
| `capture_a99.py` | Run policy, save per-tick `.npz`. |
| `viz_common.py` | Shared fkvizsim primitives (palette, `draw_arena`, fonts, fault labels). |
| `render_a99.py` | Single-pane clip, fkvizsim style. |
| `split_screen.py` | Baseline-vs-mitigation, fkvizsim style. |
| `symbol_draw.py` | Shared symbol-style arena drawer + glyph loader + final tuning constants. |
| `render_symbols.py` | Single-pane clip, **symbol style**. |
| `split_screen_symbols.py` | Baseline-vs-mitigation, **symbol style**. |
| `symbols/make_cutouts.py` | Background-removes the 3 source photos into RGBA glyphs. |
| `plot_delivery.py` | Delivery-over-time line plots. |
| `make_videos.sh` | Batch driver (fkvizsim style). |

## Symbol style — conventions

- **cream robot** = healthy, **copper robot** = faulty (no rings/halos).
- MARL action shown by (a) the **trail colour** and (b) a small **action-colour dot**
  low on the robot's right edge (grey = other/no-action; stays visible under a crate).
- **Wooden crates** draw on top of robots and lift up when carried (reads as "held").
- Final glyph sizes: `ROBOT_WU=42`, `BOX_WU=40` (in `symbol_draw.py`).
- `render_symbols.py` keeps its own inline copy of this drawing logic — if you change
  the look, update **both** it and `symbol_draw.py`.

## Run examples

```bash
PY=~/.venv/marl/bin/python

# capture (mitigation) and a matched baseline
$PY capture_a99.py --fault_type 8 --num_faults 3 --seed 708 --steps 200 --out /tmp/a99_out/F4_tick.npz
$PY capture_a99.py --fault_type 8 --num_faults 3 --seed 708 --baseline --steps 300 --out /tmp/a99_out/F4_base_full.npz

# symbol single clip + split-screen
$PY render_symbols.py --npz /tmp/a99_out/F4_tick.npz --speed 5 --out F4.mp4
$PY split_screen_symbols.py --baseline_npz /tmp/a99_out/F4_base_full.npz \
    --mitigation_npz /tmp/a99_out/F4_tick.npz --speed 5 --tail_speed 20 \
    --out split_F4.mp4 --still split_F4_final.png
```

## Outputs / delivery

- Local: `conference_videos/symbols/` (single clips), `symbols/splitscreen/`,
  `symbols/1080p/`.
- 1080p 16:9 loop versions (white-padded, 1 s freeze at start/end) via ffmpeg
  `scale=-2:1080,pad=1920:1080:...:white,tpad=...`.
- OneDrive: `Swarms/ants_presentation/symbols/{,splitscreen/,1080p/}`, files
  timestamped `_YYYYMMDD_HHMMSS` (avoids OneDrive's stale-preview cache on overwrite).
  Sync: `onedrive --sync --single-directory 'Swarms/ants_presentation' --upload-only`.
