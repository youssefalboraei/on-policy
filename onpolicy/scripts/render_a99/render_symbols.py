#!/usr/bin/env python
"""PROTOTYPE: render a captured a99 trajectory using the hand-drawn symbol images
(symbols/robot_healthy.png, robot_faulty.png, box.png) as the robot and box glyphs
instead of the flat circles/pentagons/squares.

Keeps everything else from the fkvizsim style: white bg, green deposit band,
action-coloured fading trails, action legend, honest real-time clock, side panel.

Semantics preserved on top of the cute icons:
  - faulty robots use the COPPER robot drawing (+ a bold red ring); healthy use the
    CREAM drawing.
  - the current MARL action still shows as a coloured halo ring behind each robot
    (no ring on action 0 / no-action).
  - boxes are the wooden crate; drawn UNDER robots so a carried crate reads as the
    robot sitting on its box, while free crates on the floor show in full.
"""
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import viz_common as V

SYM_DIR = os.path.join(os.path.dirname(__file__), "symbols")

# world-units footprint of each glyph (robot circle is 2*ROBOT_R = 25 wu)
ROBOT_WU = 42.0
BOX_WU = 40.0
CARRY_LIFT = 0.45        # carried crate is lifted this * BOX_WU upward (screen px)


def _load_glyph(name):
    return Image.open(os.path.join(SYM_DIR, name)).convert("RGBA")


def _fit(glyph, target_px):
    """Resize an RGBA glyph so its LONGEST side is target_px, keep aspect."""
    w, h = glyph.size
    s = target_px / max(w, h)
    return glyph.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def render(npz_path, out_path, fps=50, scale=2, speed=12.0, trail_seconds=3.0,
           end_hold=1.0):
    A = V.load(npz_path)
    W, H = A["W"], A["H"]
    rx, ry, bx, by = A["rx"], A["ry"], A["bx"], A["by"]
    acts, drate = A["actions"], A["drate"]
    faulted, tps = A["faulted"], A["tps"]
    T, nrob, nbox = A["T"], A["nrob"], A["nbox"]
    nf, ft, baseline = A["nf"], A["ft"], A["baseline"]

    stride = max(1, int(round(speed * tps / fps)))
    frame_ticks = list(range(0, T, stride))
    if frame_ticks[-1] != T - 1:
        frame_ticks.append(T - 1)
    frame_ticks += [T - 1] * int(round(end_hold * fps))
    trail_ticks = int(trail_seconds * tps)
    real_speed = fps * stride / tps

    LEG_W, PAD, TOP = 320, 10, 35
    cap_w = int((W + PAD + LEG_W) * scale)
    cap_h = int((TOP + H) * scale)
    cap_w += cap_w % 2
    cap_h += cap_h % 2

    f_time = V.font("DejaVuSans.ttf", int(18 * scale))
    f_id = V.font("DejaVuSans.ttf", int(10 * scale))
    f_leg = V.font("DejaVuSans.ttf", int(14 * scale))
    f_bold = V.font("DejaVuSans-Bold.ttf", int(15 * scale))

    # pre-resize glyphs to pixel footprint
    g_heal = _fit(_load_glyph("robot_healthy.png"), int(ROBOT_WU * scale))
    g_falt = _fit(_load_glyph("robot_faulty.png"), int(ROBOT_WU * scale))
    g_box = _fit(_load_glyph("box.png"), int(BOX_WU * scale))

    def S(v):
        return v * scale

    ax0, ay0 = 0.0, TOP

    def wx(x):
        return S(ax0 + x)

    def wy(y):
        return S(ay0 + (H - y))

    def paste_centered(img, glyph, cx, cy):
        img.alpha_composite(glyph, (int(cx - glyph.width / 2),
                                    int(cy - glyph.height / 2)))

    def draw_legend(dr):
        lx0 = W + PAD
        c2 = 50
        for label, col in V.ACTION_LEGEND:
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
        img = Image.new("RGBA", (cap_w, cap_h), (255, 255, 255, 255))
        dr = ImageDraw.Draw(img)

        t = ti / tps
        label = f"Time is {t:0.1f}s"
        tw = dr.textlength(label, font=f_time)
        dr.text((S(W / 2) - tw / 2, S(6)), label, font=f_time, fill=(0x19, 0x19, 0x19))

        dz_h = H - V.DEPOSIT_Y
        dr.rectangle([wx(0), wy(V.DEPOSIT_Y + dz_h), wx(W), wy(V.DEPOSIT_Y)],
                     fill=V.DEPOSIT)
        dr.rectangle([wx(0), wy(H), wx(W), wy(0)], outline=(0, 0, 0),
                     width=max(1, int(2 * scale)))

        # trails
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

        # robots first; the action is read from the trail colour only (no halo).
        # faulty robots are the copper drawing, healthy the cream one.
        rr = S(ROBOT_WU / 2)
        dot_r = S(ROBOT_WU * 0.135)
        dot_dx = S(ROBOT_WU * 0.24)          # right side, low enough a carried
        dot_dy = S(ROBOT_WU * 0.04)          # crate (lifted up) won't cover it
        for i in range(nrob):
            cx, cy = wx(rx[ti, i]), wy(ry[ti, i])
            paste_centered(img, g_falt if i in faulted else g_heal, cx, cy)
            # small action-colour dot (grey = other / no-action, matches legend)
            dcx, dcy = cx + dot_dx, cy - dot_dy
            dr.ellipse([dcx - dot_r, dcy - dot_r, dcx + dot_r, dcy + dot_r],
                       fill=V.action_color(int(acts[ti, i])),
                       outline=(0x1f, 0x1f, 0x1f), width=max(1, int(1.4 * scale)))
            rid = i + 1
            tx = cx - rr - S(10) if rx[ti, i] > W - 19 else cx + rr + S(2)
            ty = cy - rr - S(4) if ry[ti, i] < V.ROBOT_R + 10 else cy
            dr.text((tx, ty), f"r{rid}", font=f_id, fill=V.STROKE_ROBOT)

        # boxes ON TOP of robots; a carried crate (sitting on a robot) is lifted
        # upward so it reads as the robot holding the box above itself.
        carry_r = ROBOT_WU * 0.7
        for j in range(nbox):
            cx, cy = bx[ti, j], by[ti, j]
            if np.isnan(cx) or np.isnan(cy):
                continue
            d2 = (rx[ti] - cx) ** 2 + (ry[ti] - cy) ** 2
            carried = bool(np.nanmin(d2) <= carry_r ** 2)
            lift = S(CARRY_LIFT * BOX_WU) if carried else 0
            paste_centered(img, g_box, wx(cx), wy(cy) - lift)

        draw_legend(dr)

        lx = S(W + PAD + 60)
        ybase = TOP + 50 + 5 * 40 + 16
        dr.text((lx, S(ybase)), V.fault_label(ft, nf), font=f_bold,
                fill=(0x1f, 0x1f, 0x1f))
        dr.text((lx, S(ybase + 28)), f"Faulty robots: {nf} / {nrob}",
                font=f_leg, fill=(0x2b, 0x2b, 0x2b))
        cond = "No mitigation (baseline)" if baseline else "With mitigation"
        cond_col = V.COND_RED if baseline else V.COND_GREEN
        dr.text((lx, S(ybase + 54)), cond, font=f_bold, fill=cond_col)
        dr.text((lx, S(ybase + 82)), f"Delivered: {float(drate[ti]):.0f} / {nbox}",
                font=f_leg, fill=(0x2b, 0x2b, 0x2b))
        dr.text((lx, S(ybase + 106)), f"playback: {real_speed:.0f}x real time",
                font=f_leg, fill=(0x88, 0x88, 0x88))

        writer.append_data(np.asarray(img.convert("RGB")))

    writer.close()
    print(f"[symbols] {len(frame_ticks)} frames @ {fps}fps ({cap_w}x{cap_h}), "
          f"{real_speed:.0f}x speed, {T/tps:.0f}s sim -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--trail_seconds", type=float, default=3.0)
    ap.add_argument("--end_hold", type=float, default=1.0)
    a = ap.parse_args()
    render(a.npz, a.out, fps=a.fps, scale=a.scale, speed=a.speed,
           trail_seconds=a.trail_seconds, end_hold=a.end_hold)
