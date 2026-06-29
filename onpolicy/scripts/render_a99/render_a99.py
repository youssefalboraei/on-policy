#!/usr/bin/env python
"""Render a captured a99 PER-TICK trajectory (.npz from capture_a99.py) to MP4 in
the fkvizsim neurov canvas style (replicated from static/neurov_viz.js).

Time is real: one tick = 0.02s (50 ticks/sec). A full task is ~5 min of sim time,
so the clip is a timelapse at `--speed` x real time with an honest accelerating
`Time is {t}s` clock. Trails fade over a fixed number of real SECONDS, so their
disappearing time is constant across clips. Motion is the true physics (per tick),
not interpolation.

Style: white bg, green deposit band, blue box outlines, faulty robots as pentagons,
healthy robots as action-coloured circles (hollow on the default action), `r#`
labels, opacity-ramped action-coloured trails, action legend, centered time label.
"""
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import imageio.v2 as imageio

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
BLUE = (0x00, 0x00, 0xff)
LEG_W, PAD, TOP = 320, 10, 35
FAULT_CODE = {3: "F1", 4: "F2", 5: "F3", 8: "F4"}
FAULT_TECH = {0: "none", 3: "ALL_WHEEL_V0", 4: "ALL_WHEEL_V10",
              5: "ALL_WHEEL_V50", 8: "PICKUP"}
FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts/ttf")


def _font(name, px):
    return ImageFont.truetype(os.path.join(FONT_DIR, name), px)


def action_color(a):
    return ACTION_COLORS.get(int(a), OTHER)


def blend_on_white(rgb, alpha):
    return tuple(int(c * alpha + 255 * (1 - alpha)) for c in rgb)


def render(npz_path, out_path, fps=50, scale=2, speed=12.0, trail_seconds=3.0,
           end_hold=1.0, title=None):
    d = np.load(npz_path)
    rx, ry, bx, by = d["rx"], d["ry"], d["bx"], d["by"]
    acts, drate = d["actions"], d["drate"]
    faulted = set(int(i) for i in d["faulted"])
    W, H = float(d["arena_w"]), float(d["arena_h"])
    tps = int(d["ticks_per_sec"])
    T, nrob = rx.shape
    nbox = bx.shape[1]
    nf = int(d["num_faults"])
    ft = int(d["fault_type"])
    baseline = bool(int(d["baseline"])) if "baseline" in d else False

    # timelapse: advance `stride` ticks per output frame so video plays `speed`x
    # real time. video-sec per real-sec = fps*stride/tps -> stride = speed*tps/fps
    stride = max(1, int(round(speed * tps / fps)))
    frame_ticks = list(range(0, T, stride))
    if frame_ticks[-1] != T - 1:
        frame_ticks.append(T - 1)
    frame_ticks += [T - 1] * int(round(end_hold * fps))     # freeze on completion
    trail_ticks = int(trail_seconds * tps)
    real_speed = fps * stride / tps

    cap_w = int((W + PAD + LEG_W) * scale)
    cap_h = int((TOP + H) * scale)
    cap_w += cap_w % 2
    cap_h += cap_h % 2

    f_time = _font("DejaVuSans.ttf", int(18 * scale))
    f_id = _font("DejaVuSans.ttf", int(10 * scale))
    f_leg = _font("DejaVuSans.ttf", int(14 * scale))
    f_bold = _font("DejaVuSans-Bold.ttf", int(15 * scale))

    def S(v):
        return v * scale

    ax0, ay0 = 0.0, TOP

    def wx(x):
        return S(ax0 + x)

    def wy(y):
        return S(ay0 + (H - y))

    def draw_legend(dr):
        lx0 = W + PAD
        c2 = 50
        for label, col in ACTION_LEGEND:
            cx, cy = lx0 + 60, TOP + c2 + 10
            r = 10
            dr.ellipse([S(cx - r), S(cy - r), S(cx + r), S(cy + r)],
                       fill=col, outline=(0, 0, 0), width=max(1, int(1.5 * scale)))
            dr.text((S(lx0 + 85), S(cy) - f_leg.size * 0.5), label,
                    font=f_leg, fill=(0x2b, 0x2b, 0x2b))
            c2 += 40

    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264", macro_block_size=None,
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow"])

    for ti in frame_ticks:
        img = Image.new("RGB", (cap_w, cap_h), (255, 255, 255))
        dr = ImageDraw.Draw(img)

        t = ti / tps
        label = f"Time is {t:0.1f}s"
        tw = dr.textlength(label, font=f_time)
        dr.text((S(W / 2) - tw / 2, S(6)), label, font=f_time, fill=(0x19, 0x19, 0x19))

        dz_h = H - 425.0
        dr.rectangle([wx(0), wy(425.0 + dz_h), wx(W), wy(425.0)], fill=DEPOSIT)
        dr.rectangle([wx(0), wy(H), wx(W), wy(0)], outline=(0, 0, 0),
                     width=max(1, int(2 * scale)))

        # trails: last `trail_ticks`, sampled every `stride` ticks, fading to nothing
        lo = max(0, ti - trail_ticks)
        samples = list(range(lo, ti, stride))
        if samples and samples[-1] != ti:
            samples.append(ti)
        nseg = len(samples) - 1
        for i in range(nrob):
            for k in range(nseg):
                s0, s1 = samples[k], samples[k + 1]
                frac = (k + 1) / max(nseg, 1)            # 0 (old) .. 1 (new)
                alpha = (frac ** 2) * 0.9
                col = blend_on_white(action_color(acts[s1, i]), alpha)
                dr.line([wx(rx[s0, i]), wy(ry[s0, i]),
                         wx(rx[s1, i]), wy(ry[s1, i])],
                        fill=col, width=max(1, int(4 * scale)))

        half = BOX_SIDE / 2
        for j in range(nbox):
            cx, cy = bx[ti, j], by[ti, j]
            if np.isnan(cx) or np.isnan(cy):     # delivered -> off the floor
                continue
            dr.rectangle([wx(cx - half), wy(cy + half), wx(cx + half), wy(cy - half)],
                         outline=BLUE, width=max(1, int(2.5 * scale)))

        for i in range(nrob):
            cx, cy = wx(rx[ti, i]), wy(ry[ti, i])
            r = S(ROBOT_R)
            a = int(acts[ti, i])
            fill = action_color(a) if a != 0 else None
            lw = max(1, int(2 * scale))
            if i in faulted:
                pts = [(cx + r * np.sin(k * 2 * np.pi / 5),
                        cy - r * np.cos(k * 2 * np.pi / 5)) for k in range(5)]
                dr.polygon(pts, fill=fill, outline=STROKE_ROBOT, width=lw)
            else:
                dr.ellipse([cx - r, cy - r, cx + r, cy + r],
                           fill=fill, outline=STROKE_ROBOT, width=lw)
            rid = i + 1
            tx = cx - r - S(10) if rx[ti, i] > W - ROBOT_R - 19 else cx + r + S(1)
            ty = cy - r - S(4) if ry[ti, i] < ROBOT_R + 10 else cy
            dr.text((tx, ty), f"r{rid}", font=f_id, fill=STROKE_ROBOT)

        draw_legend(dr)

        lx = S(W + PAD + 60)
        ybase = TOP + 50 + 5 * 40 + 16
        if title:
            fault_line = title
        elif ft == 0 or nf == 0:
            fault_line = "Fault: none"
        else:
            fault_line = f"Fault: {FAULT_CODE.get(ft, 'T'+str(ft))} ({FAULT_TECH.get(ft, ft)})"
        dr.text((lx, S(ybase)), fault_line, font=f_bold, fill=(0x1f, 0x1f, 0x1f))
        dr.text((lx, S(ybase + 28)), f"Faulty robots: {nf} / {nrob}",
                font=f_leg, fill=(0x2b, 0x2b, 0x2b))
        cond = "No mitigation (baseline)" if baseline else "With mitigation"
        cond_col = (0xb0, 0x30, 0x30) if baseline else (0x2e, 0x6f, 0x4e)
        dr.text((lx, S(ybase + 54)), cond, font=f_bold, fill=cond_col)
        dr.text((lx, S(ybase + 82)), f"Delivered: {float(drate[ti]):.0f} / {nbox}",
                font=f_leg, fill=(0x2b, 0x2b, 0x2b))
        dr.text((lx, S(ybase + 106)), f"playback: {real_speed:.0f}x real time",
                font=f_leg, fill=(0x88, 0x88, 0x88))

        writer.append_data(np.asarray(img))

    writer.close()
    print(f"[render] {len(frame_ticks)} frames @ {fps}fps ({cap_w}x{cap_h}), "
          f"{real_speed:.0f}x speed, {T/tps:.0f}s sim -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--scale", type=int, default=2, help="supersample factor")
    ap.add_argument("--speed", type=float, default=12.0,
                    help="playback speed as a multiple of real time")
    ap.add_argument("--trail_seconds", type=float, default=3.0,
                    help="trail persistence in real SIM seconds (constant fade time)")
    ap.add_argument("--end_hold", type=float, default=1.0,
                    help="freeze the final frame for this many video seconds")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    render(a.npz, a.out, fps=a.fps, scale=a.scale, speed=a.speed,
           trail_seconds=a.trail_seconds, end_hold=a.end_hold, title=a.title)
