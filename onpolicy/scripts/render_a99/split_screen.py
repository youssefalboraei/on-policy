#!/usr/bin/env python
"""Side-by-side baseline (no mitigation) vs mitigation, one shared clock.

Both inputs must be the same scenario at matched duration (capture the baseline
with --fixed_ticks = the mitigation run's tick count). Renders a 16:9-ish slide
clip and saves the final frame as a PNG still.
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import viz_common as V

# layout (unscaled world px)
MARGIN = 24
HEADER = 58          # top fault/time bar
SUBTITLE = 40        # per-arena condition title
DELIV = 40           # per-arena delivered line
FOOTER = 50          # legend strip
GAP = 48             # between arenas


def render(left_npz, right_npz, out_path, fps=50, scale=2, speed=12.0,
           tail_speed=20.0, trail_seconds=3.0, end_hold=1.2, still_path=None):
    L, R = V.load(left_npz), V.load(right_npz)
    W, H = L["W"], L["H"]
    tps = L["tps"]
    T = max(L["T"], R["T"])          # run to whichever side finishes last
    ft, nf = R["ft"], R["nf"]
    nbox = R["nbox"]

    def completion_tick(A):
        hit = np.where(A["drate"] >= nbox)[0]
        return int(hit[0]) if len(hit) else None

    Lc, Rc = completion_tick(L), completion_tick(R)

    # Variable speed: play at `speed` during the active race, then `tail_speed`
    # once the first side has finished, while the slower side catches up.
    stride1 = max(1, int(round(speed * tps / fps)))
    stride2 = max(1, int(round(tail_speed * tps / fps)))
    comps = [c for c in (Lc, Rc) if c is not None]
    switch = min(comps) if comps else T
    seq = []                                       # (tick, speed_for_this_frame)
    t = 0
    while t < switch:
        seq.append((t, speed)); t += stride1
    t = switch
    while t < T:
        seq.append((t, tail_speed)); t += stride2
    if not seq or seq[-1][0] != T - 1:
        seq.append((T - 1, tail_speed if switch < T else speed))
    seq += [(T - 1, tail_speed if switch < T else speed)] * int(round(end_hold * fps))
    trail_ticks = int(trail_seconds * tps)

    arena_top = HEADER + SUBTITLE
    cap_w = int((2 * W + GAP + 2 * MARGIN) * scale)
    cap_h = int((arena_top + H + DELIV + FOOTER + MARGIN) * scale)
    cap_w += cap_w % 2
    cap_h += cap_h % 2

    f_hdr = V.font("DejaVuSans-Bold.ttf", 22 * scale)
    f_sub = V.font("DejaVuSans-Bold.ttf", 22 * scale)
    f_del = V.font("DejaVuSans-Bold.ttf", 20 * scale)
    f_id = V.font("DejaVuSans.ttf", 10 * scale)
    f_leg = V.font("DejaVuSans.ttf", 14 * scale)

    lx = MARGIN
    rx0 = MARGIN + W + GAP

    def centered(dr, text, font, cx, y, fill):
        w = dr.textlength(text, font=font)
        dr.text((cx * scale - w / 2, y * scale), text, font=font, fill=fill)

    def compose(ti, spd):
        img = Image.new("RGB", (cap_w, cap_h), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        # header: fault | faulty | time | speed (speed rises during catch-up)
        t = ti / tps
        fault_txt = V.fault_label(ft, nf).replace("Fault: ", "")
        spd_txt = f"{spd:.0f}x real time"
        hdr = (f"Fault: {fault_txt}      " +
               (f"{nf} / {L['nrob']} faulty      " if nf > 0 else "") +
               f"Time {t:0.1f}s      {spd_txt}")
        centered(dr, hdr, f_hdr, (2 * W + GAP + 2 * MARGIN) / 2, 16, (0x19, 0x19, 0x19))

        # arena titles
        centered(dr, "NO MITIGATION (baseline)", f_sub, lx + W / 2,
                 HEADER + 6, V.COND_RED)
        centered(dr, "WITH MITIGATION", f_sub, rx0 + W / 2,
                 HEADER + 6, V.COND_GREEN)

        # each side freezes at its final frame once it has finished
        li = min(ti, L["T"] - 1)
        ri = min(ti, R["T"] - 1)

        # arenas (sample trails at the base stride for consistent density)
        V.draw_arena(dr, L, li, lx, arena_top, scale, stride1, trail_ticks, f_id)
        V.draw_arena(dr, R, ri, rx0, arena_top, scale, stride1, trail_ticks, f_id)

        # delivered lines under each (+ own completion time once done)
        def deliv_text(A, idx, comp):
            d = float(A["drate"][idx])
            if comp is not None and idx >= comp:
                return f"Delivered: {d:.0f} / {nbox}  (done at {comp / tps:.0f}s)"
            return f"Delivered: {d:.0f} / {nbox}"
        ldone = Lc is not None and li >= Lc
        rdone = Rc is not None and ri >= Rc
        centered(dr, deliv_text(L, li, Lc), f_del, lx + W / 2, arena_top + H + 8,
                 V.COND_GREEN if ldone else (0x2b, 0x2b, 0x2b))
        centered(dr, deliv_text(R, ri, Rc), f_del, rx0 + W / 2, arena_top + H + 8,
                 V.COND_GREEN if rdone else (0x2b, 0x2b, 0x2b))

        # footer legend (centered)
        n = len(V.ACTION_LEGEND)
        gap = 190
        total = gap * (n - 1) + 120
        x0 = (2 * W + GAP + 2 * MARGIN) / 2 - total / 2
        V.legend_strip(dr, x0, arena_top + H + DELIV + 4, scale, f_leg, gap=gap)
        return img

    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264", macro_block_size=None,
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p", "-preset", "slow"])
    last = None
    for ti, spd in seq:
        last = compose(ti, spd)
        writer.append_data(np.asarray(last))
    writer.close()
    if still_path and last is not None:
        last.save(still_path)
    sw_s = switch / tps if switch < T else None
    print(f"[split] {len(seq)} frames ({cap_w}x{cap_h}), {speed:.0f}x->"
          f"{tail_speed:.0f}x" + (f" @ {sw_s:.0f}s" if sw_s else "") +
          f", {T/tps:.0f}s sim -> {out_path}" +
          (f"  still -> {still_path}" if still_path else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_npz", required=True)
    ap.add_argument("--mitigation_npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--still", default=None)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--speed", type=float, default=5.0,
                    help="playback speed during the active race")
    ap.add_argument("--tail_speed", type=float, default=20.0,
                    help="faster playback once the first side finishes (catch-up)")
    ap.add_argument("--trail_seconds", type=float, default=3.0)
    a = ap.parse_args()
    render(a.baseline_npz, a.mitigation_npz, a.out, fps=a.fps, scale=a.scale,
           speed=a.speed, tail_speed=a.tail_speed, trail_seconds=a.trail_seconds,
           still_path=a.still)
