#!/usr/bin/env python
"""Shared fkvizsim-style drawing primitives for the a99 video pipeline.

Used by render_a99.py (single clip), split_screen.py (baseline vs mitigation) and
plot_delivery.py so every artifact shares one visual language.
"""
import os
import numpy as np
from PIL import ImageFont
import matplotlib

OTHER = (0x66, 0x66, 0x66)
ACTION_COLORS = {3: (0xdf, 0x4d, 0x54), 4: (0xFF, 0xD2, 0x66),
                 9: (0xA6, 0x89, 0xC2), 6: (0xa8, 0x67, 0x4a)}
ACTION_LEGEND = [
    ("Bias to nearest robot", ACTION_COLORS[3]),
    ("Bias to nearest box",   ACTION_COLORS[4]),
    ("Bias from nearest wall", ACTION_COLORS[9]),
    ("Bias left",             ACTION_COLORS[6]),
    ("Other actions",         OTHER),
]
ROBOT_R = 12.5
BOX_SIDE = 12.5 * 1.9
DEPOSIT = (0x54, 0x91, 0x54)
STROKE_ROBOT = (0x1f, 0x1f, 0x1f)
FAULT_STROKE = (0xd0, 0x10, 0x10)   # bright red outline for faulty robots
BLUE = (0x00, 0x00, 0xff)
DEPOSIT_Y = 425.0
FAULT_CODE = {3: "F1", 4: "F2", 5: "F3", 8: "F4"}
FAULT_TECH = {0: "none", 3: "ALL_WHEEL_V0", 4: "ALL_WHEEL_V10",
              5: "ALL_WHEEL_V50", 8: "PICKUP"}
COND_GREEN = (0x2e, 0x6f, 0x4e)
COND_RED = (0xb0, 0x30, 0x30)
FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts/ttf")


def font(name, px):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), int(px))


def action_color(a):
    return ACTION_COLORS.get(int(a), OTHER)


def blend_on_white(rgb, alpha):
    return tuple(int(c * alpha + 255 * (1 - alpha)) for c in rgb)


def fault_label(ft, nf):
    if ft == 0 or nf == 0:
        return "Fault: none"
    return f"Fault: {FAULT_CODE.get(ft, 'T'+str(ft))} ({FAULT_TECH.get(ft, ft)})"


def load(npz_path):
    """Load a capture .npz into a plain dict with a `faulted` set."""
    d = np.load(npz_path)
    A = {k: d[k] for k in d.files}
    A["faulted"] = set(int(i) for i in d["faulted"])
    A["W"], A["H"] = float(d["arena_w"]), float(d["arena_h"])
    A["tps"] = int(d["ticks_per_sec"])
    A["nf"], A["ft"] = int(d["num_faults"]), int(d["fault_type"])
    A["baseline"] = bool(int(d["baseline"])) if "baseline" in d.files else False
    A["T"], A["nrob"] = A["rx"].shape
    A["nbox"] = A["bx"].shape[1]
    return A


def draw_arena(dr, A, ti, ox, oy, scale, stride, trail_ticks, f_id):
    """Draw one fkvizsim arena (deposit, border, trails, boxes, robots, ids) at
    world-offset (ox, oy) in capture px / `scale`."""
    W, H = A["W"], A["H"]
    rx, ry, bx, by, acts = A["rx"], A["ry"], A["bx"], A["by"], A["actions"]
    faulted = A["faulted"]
    nrob, nbox = A["nrob"], A["nbox"]

    def wx(x):
        return (ox + x) * scale

    def wy(y):
        return (oy + (H - y)) * scale

    dz_h = H - DEPOSIT_Y
    dr.rectangle([wx(0), wy(DEPOSIT_Y + dz_h), wx(W), wy(DEPOSIT_Y)], fill=DEPOSIT)
    dr.rectangle([wx(0), wy(H), wx(W), wy(0)], outline=(0, 0, 0),
                 width=max(1, int(2 * scale)))

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
            col = blend_on_white(action_color(acts[s1, i]), alpha)
            dr.line([wx(rx[s0, i]), wy(ry[s0, i]), wx(rx[s1, i]), wy(ry[s1, i])],
                    fill=col, width=max(1, int(4 * scale)))

    for i in range(nrob):
        cx, cy = wx(rx[ti, i]), wy(ry[ti, i])
        r = ROBOT_R * scale
        a = int(acts[ti, i])
        fill = action_color(a) if a != 0 else None
        lw = max(1, int(2 * scale))
        if i in faulted:
            pts = [(cx + r * np.sin(k * 2 * np.pi / 5),
                    cy - r * np.cos(k * 2 * np.pi / 5)) for k in range(5)]
            dr.polygon(pts, fill=fill, outline=FAULT_STROKE,
                       width=max(2, int(3.4 * scale)))
        else:
            dr.ellipse([cx - r, cy - r, cx + r, cy + r],
                       fill=fill, outline=STROKE_ROBOT, width=lw)
        rid = i + 1
        tx = cx - r - 10 * scale if rx[ti, i] > W - ROBOT_R - 19 else cx + r + 1 * scale
        ty = cy - r - 4 * scale if ry[ti, i] < ROBOT_R + 10 else cy
        dr.text((tx, ty), f"r{rid}", font=f_id, fill=STROKE_ROBOT)

    # boxes drawn on top so a carried box reads as a bold square framing its robot
    half = max(BOX_SIDE / 2, ROBOT_R + 4)
    for j in range(nbox):
        cx, cy = bx[ti, j], by[ti, j]
        if np.isnan(cx) or np.isnan(cy):
            continue
        dr.rectangle([wx(cx - half), wy(cy + half), wx(cx + half), wy(cy - half)],
                     outline=BLUE, width=max(2, int(3.2 * scale)))


def legend_strip(dr, x0, y0, scale, f_leg, horizontal=True, gap=190):
    """Draw the action legend as a horizontal strip (for split-screen footer)."""
    x = x0
    for label, col in ACTION_LEGEND:
        r = 9
        cy = y0 + 10
        dr.ellipse([(x) * scale, (cy - r) * scale, (x + 2 * r) * scale, (cy + r) * scale],
                   fill=col, outline=(0, 0, 0), width=max(1, int(1.2 * scale)))
        dr.text(((x + 2 * r + 6) * scale, (cy - 8) * scale), label, font=f_leg,
                fill=(0x2b, 0x2b, 0x2b))
        x += gap
