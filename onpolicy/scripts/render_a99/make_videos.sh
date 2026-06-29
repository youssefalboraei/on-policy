#!/usr/bin/env bash
# Generate a99 (ST) ST-SE policy videos for the conference talk.
# Capture a live episode per (fault, seed) then render a slow-mo MP4.
#
# Usage:  ./make_videos.sh [OUTDIR]
# Requires the marl venv (torch + marl_sim) and that marl_sim.so is built with
# the write_viz binding (see CLAUDE.md build notes).
set -euo pipefail

PY="$HOME/.venv/marl/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"          # on-policy-eval root
OUT="${1:-$REPO/../conference_videos}"
TMP="$(mktemp -d)"
mkdir -p "$OUT"

# (name  fault_type  num_faults  seed)
RUNS=(
  "F1_ALL_WHEEL_V0  3  3  708"
  "F4_PICKUP        8  3  708"
  "NoFault          0  0  708"
  # add more, e.g. heavier failure for a longer/harder story:
  # "F1_5frozen     3  5  708"
  # "F3_ALL_WHEEL_V50 5 3 708"
)

cd "$REPO"
for r in "${RUNS[@]}"; do
  read -r name ft nf seed <<<"$r"
  npz="$TMP/${name}.npz"
  bnpz="$TMP/${name}_baseline.npz"
  echo "=== $name (ft=$ft nf=$nf seed=$seed) : mitigation ==="
  PYTHONPATH=. "$PY" onpolicy/scripts/render_a99/capture_a99.py \
      --fault_type "$ft" --num_faults "$nf" --seed "$seed" --steps 400 --out "$npz"
  "$PY" onpolicy/scripts/render_a99/render_a99.py \
      --npz "$npz" --out "$OUT/a99_ST-SE_${name}.mp4" --speed 12 --trail_seconds 3

  # baseline (no mitigation): same seed/fault, matched to the mitigation duration
  TICKS=$("$PY" -c "import numpy as np; print(np.load('$npz')['rx'].shape[0])")
  echo "=== $name : baseline (no mitigation, $TICKS ticks) ==="
  PYTHONPATH=. "$PY" onpolicy/scripts/render_a99/capture_a99.py \
      --baseline --fault_type "$ft" --num_faults "$nf" --seed "$seed" \
      --steps 400 --fixed_ticks "$TICKS" --out "$bnpz"
  "$PY" onpolicy/scripts/render_a99/render_a99.py \
      --npz "$bnpz" --out "$OUT/a99_ST-SE_${name}_baseline.mp4" --speed 12 --trail_seconds 3
done
echo "Done -> $OUT"
