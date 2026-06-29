#!/usr/bin/env python
"""Shared symbol-style arena drawer (hand-drawn robot/box glyphs).

Mirrors the FINAL look approved for render_symbols.py so the single clips and the
split-screens share one visual language:
  - cream robot = healthy, copper robot = faulty (no rings/halos)
  - MARL action shown by the trail colour AND a small action-colour dot placed low
    on the robot's right edge (grey = other/no-action; stays visible under a crate)
  - wooden crate boxes draw ON TOP of robots and lift up when carried
"""
import os
import numpy as np
from PIL import Image
import viz_common as V

SYM_DIR = os.path.join(os.path.dirname(__file__), "symbols")
ROBOT_WU = 42.0
BOX_WU = 40.0
CARRY_LIFT = 0.45         # carried crate lifts this * BOX_WU upward
DOT_R = 0.135             # * ROBOT_WU
DOT_DX = 0.24             # * ROBOT_WU, right of centre
DOT_DY = 0.04             # * ROBOT_WU, just below centre
CARRY_R = 0.7             # box within this * ROBOT_WU of a robot reads as carried


def load_glyphs(scale):
    def fit(g, t):
        w, h = g.size
        s = t / max(w, h)
        return g.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)

    def ld(n):
        return Image.open(os.path.join(SYM_DIR, n)).convert("RGBA")

    return {
        "heal": fit(ld("robot_healthy.png"), int(ROBOT_WU * scale)),
        "falt": fit(ld("robot_faulty.png"), int(ROBOT_WU * scale)),
        "box": fit(ld("box.png"), int(BOX_WU * scale)),
    }


def draw_arena_symbols(img, dr, A, ti, ox, oy, scale, stride, trail_ticks, f_id,
                       glyphs):
    """Symbol-style twin of viz_common.draw_arena. `img` is the RGBA canvas the
    glyphs alpha-composite onto; `dr` is its ImageDraw for the vector bits."""
    W, H = A["W"], A["H"]
    rx, ry, bx, by, acts = A["rx"], A["ry"], A["bx"], A["by"], A["actions"]
    faulted = A["faulted"]
    nrob, nbox = A["nrob"], A["nbox"]

    def wx(x):
        return (ox + x) * scale

    def wy(y):
        return (oy + (H - y)) * scale

    def pc(glyph, cx, cy):
        img.alpha_composite(glyph, (int(cx - glyph.width / 2),
                                    int(cy - glyph.height / 2)))

    dz_h = H - V.DEPOSIT_Y
    dr.rectangle([wx(0), wy(V.DEPOSIT_Y + dz_h), wx(W), wy(V.DEPOSIT_Y)],
                 fill=V.DEPOSIT)
    dr.rectangle([wx(0), wy(H), wx(W), wy(0)], outline=(0, 0, 0),
                 width=max(1, int(2 * scale)))

    # trails (action-coloured, fading)
    lo = max(0, ti - trail_ticks)
    samples = list(range(lo, ti, stride))
    if samples and samples[-1] != ti:
        samples.append(ti)
    nseg = len(samples) - 1
    for i in range(nrob):
        for k in range(nseg):
            s0, s1 = samples[k], samples[k + 1]
            frac = (k + 1) / max(nseg, 1)
            alpha = (frac ** 2) * 0.9
            col = V.blend_on_white(V.action_color(acts[s1, i]), alpha)
            dr.line([wx(rx[s0, i]), wy(ry[s0, i]), wx(rx[s1, i]), wy(ry[s1, i])],
                    fill=col, width=max(1, int(4 * scale)))

    # robots + action dot
    rr = ROBOT_WU / 2 * scale
    dot_r = ROBOT_WU * DOT_R * scale
    dot_dx = ROBOT_WU * DOT_DX * scale
    dot_dy = ROBOT_WU * DOT_DY * scale
    for i in range(nrob):
        cx, cy = wx(rx[ti, i]), wy(ry[ti, i])
        pc(glyphs["falt"] if i in faulted else glyphs["heal"], cx, cy)
        dcx, dcy = cx + dot_dx, cy - dot_dy
        dr.ellipse([dcx - dot_r, dcy - dot_r, dcx + dot_r, dcy + dot_r],
                   fill=V.action_color(int(acts[ti, i])),
                   outline=(0x1f, 0x1f, 0x1f), width=max(1, int(1.4 * scale)))
        rid = i + 1
        tx = cx - rr - 10 * scale if rx[ti, i] > W - 19 else cx + rr + 2 * scale
        ty = cy - rr - 4 * scale if ry[ti, i] < V.ROBOT_R + 10 else cy
        dr.text((tx, ty), f"r{rid}", font=f_id, fill=V.STROKE_ROBOT)

    # boxes ON TOP; carried crate lifts up so it reads as being held
    carry_r2 = (ROBOT_WU * CARRY_R) ** 2
    for j in range(nbox):
        cx, cy = bx[ti, j], by[ti, j]
        if np.isnan(cx) or np.isnan(cy):
            continue
        d2 = (rx[ti] - cx) ** 2 + (ry[ti] - cy) ** 2
        lift = CARRY_LIFT * BOX_WU * scale if float(np.nanmin(d2)) <= carry_r2 else 0
        pc(glyphs["box"], wx(cx), wy(cy) - lift)
