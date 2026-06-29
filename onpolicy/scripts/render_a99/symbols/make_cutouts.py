#!/usr/bin/env python
"""Remove the flat grey background from the three hand-drawn symbol photos and
crop to the inked object, producing tight RGBA PNGs for use as viz glyphs.

Strategy: flood-fill inward from every border pixel with a colour tolerance.
The object is fully enclosed by a dark ink outline, so the fill can't leak in.
That handles both the uniform grey card AND the soft grey shadow behind the box
(filled in a second, looser pass on near-grey pixels the flood missed)."""
import sys
import numpy as np
from PIL import Image
from collections import deque

SRC = {
    "robot_healthy": "fc05e725-IMG_7505.jpeg",   # cream robot
    "robot_faulty":  "46014620-IMG_7506.jpeg",    # copper robot
    "box":           "4be95a60-IMG_7504.jpeg",    # wooden crate
}
UPLOADS = "/home/kg23671/.claude/uploads/66dd7422-a3c2-483c-90a0-275a93208dbe"
OUT = "/home/kg23671/projects/mcs_codebase/on-policy-eval/onpolicy/scripts/render_a99/symbols"


def flood_alpha(rgb, tol=42):
    """Return a bool mask of background pixels reachable from the border."""
    h, w, _ = rgb.shape
    bg = np.zeros((h, w), bool)
    seen = np.zeros((h, w), bool)
    dq = deque()
    # seed from a robust border colour estimate (median of the 4 corners region)
    corners = np.concatenate([
        rgb[:8, :8].reshape(-1, 3), rgb[:8, -8:].reshape(-1, 3),
        rgb[-8:, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3)])
    ref = np.median(corners, axis=0)
    for x in range(w):
        for y in (0, h - 1):
            dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            dq.append((y, x))
    tol2 = tol * tol
    while dq:
        y, x = dq.popleft()
        if seen[y, x]:
            continue
        seen[y, x] = True
        d = rgb[y, x].astype(int) - ref
        if d.dot(d) > tol2:
            continue
        bg[y, x] = True
        if y > 0:
            dq.append((y - 1, x))
        if y < h - 1:
            dq.append((y + 1, x))
        if x > 0:
            dq.append((y, x - 1))
        if x < w - 1:
            dq.append((y, x + 1))
    return bg


def cutout(name, fname):
    im = Image.open(f"{UPLOADS}/{fname}").convert("RGB")
    rgb = np.asarray(im)
    bg = flood_alpha(rgb)
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    # soft-feather the 1px edge so the ink doesn't get a hard grey halo
    out = np.dstack([rgb, alpha])
    rgba = Image.fromarray(out, "RGBA")
    # crop to opaque bbox
    ys, xs = np.where(alpha > 0)
    pad = 6
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad),
           min(rgba.width, xs.max() + pad), min(rgba.height, ys.max() + pad))
    rgba = rgba.crop(box)
    rgba.save(f"{OUT}/{name}.png")
    print(f"{name}: {rgba.size}  (bg pixels removed: {bg.sum()})")


if __name__ == "__main__":
    for name, fname in SRC.items():
        cutout(name, fname)
