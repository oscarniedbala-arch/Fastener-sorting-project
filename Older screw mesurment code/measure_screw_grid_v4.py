#!/usr/bin/env python3
import argparse
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


@dataclass
class ScrewMeasurement:
    id: int
    center_px: Tuple[float, float]
    length_overall_mm: float
    length_underhead_mm: float
    diameter_mm: float
    m_guess: str


def load_image(path_or_url: str) -> Optional[np.ndarray]:
    if path_or_url.lower().startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url, timeout=10) as r:
            data = r.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.imread(path_or_url, cv2.IMREAD_COLOR)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def auto_crop_camera_region(bgr: np.ndarray) -> np.ndarray:
    """Crop UI screenshots to largest dark-ish camera pane; if already a frame, returns near-original."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w * h < 0.25 * bgr.shape[0] * bgr.shape[1]:
        return bgr
    return bgr[y:y + h, x:x + w]


def detect_grid_squares(gray: np.ndarray, bright_thresh: int) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Detect bright square holes and return contours and centers."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(g, bright_thresh, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares, centers = [], []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 250 or area > 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8:
            continue
        ar = w / float(h)
        if ar < 0.7 or ar > 1.3:
            continue
        squares.append(c)
        centers.append((x + w / 2.0, y + h / 2.0))
    return squares, centers


def estimate_pitch_px_nn(centers: List[Tuple[float, float]]) -> Optional[float]:
    """Global pitch estimate: median nearest-neighbor distance among centers."""
    if len(centers) < 20:
        return None
    pts = np.array(centers, dtype=np.float32)
    dmins = []
    for i in range(len(pts)):
        d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
        d = d[d > 1e-3]
        dmins.append(float(d.min()))
    pitch = float(np.median(dmins))
    if pitch < 10 or pitch > 300:
        return None
    return pitch


def crop_to_grid(gray: np.ndarray, centers: List[Tuple[float, float]], pad_px: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    pts = np.array(centers, dtype=np.float32)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    x0 = int(clamp(xmin - pad_px, 0, gray.shape[1] - 1))
    y0 = int(clamp(ymin - pad_px, 0, gray.shape[0] - 1))
    x1 = int(clamp(xmax + pad_px, 1, gray.shape[1]))
    y1 = int(clamp(ymax + pad_px, 1, gray.shape[0]))
    return gray[y0:y1, x0:x1], (x0, y0)


def rotate_patch(gray: np.ndarray, rect, pad: int = 55) -> np.ndarray:
    """Rotate so screw axis is vertical, then crop a patch."""
    (cx, cy), (w, h), ang = rect
    if w < h:
        rot_angle = ang
        long_side = h
        short_side = w
    else:
        rot_angle = ang + 90.0
        long_side = w
        short_side = h

    M = cv2.getRotationMatrix2D((cx, cy), rot_angle, 1.0)
    rot = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR)

    bw = int(short_side + 2 * pad)
    bh = int(long_side + 2 * pad)
    x0 = int(cx - bw / 2)
    y0 = int(cy - bh / 2)

    x0 = clamp(x0, 0, gray.shape[1] - 1)
    y0 = clamp(y0, 0, gray.shape[0] - 1)
    x1 = clamp(x0 + bw, 1, gray.shape[1])
    y1 = clamp(y0 + bh, 1, gray.shape[0])

    patch = rot[y0:y1, x0:x1]
    if patch.shape[1] > patch.shape[0]:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    return patch


def width_profile_contiguous_run(patch: np.ndarray, search: int = 14) -> Tuple[np.ndarray, float]:
    """
    Compute per-row width using contiguous dark run near the screw axis.
    Returns widths (px) for each row and the Otsu threshold used.
    """
    p = np.minimum(patch.astype(np.uint8), 220)
    p = cv2.GaussianBlur(p, (3, 3), 0)
    thr, _ = cv2.threshold(p, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thr = float(thr)

    h, w = p.shape
    axis = w // 2
    widths = np.zeros(h, dtype=np.float32)

    for r in range(h):
        lo = max(0, axis - search)
        hi = min(w, axis + search + 1)
        row = p[r, lo:hi]
        c = int(np.argmin(row)) + lo
        if p[r, c] >= thr:
            continue

        l = c
        while l > 0 and p[r, l] < thr:
            l -= 1
        rr = c
        while rr < w - 1 and p[r, rr] < thr:
            rr += 1
        widths[r] = (rr - l - 1)

    return widths, thr


def longest_good_run(widths: np.ndarray, min_w_px: int) -> Optional[Tuple[int, int]]:
    good = widths >= float(min_w_px)
    h = len(widths)
    best_len = 0
    best = None
    i = 0
    while i < h:
        if not good[i]:
            i += 1
            continue
        j = i
        while j < h and good[j]:
            j += 1
        if (j - i) > best_len:
            best_len = (j - i)
            best = (i, j)
        i = j
    return best if best_len >= 20 else None


def estimate_underhead_length_px(widths: np.ndarray, run: Tuple[int, int], head_ratio: float = 1.35) -> Tuple[float, float]:
    """
    Given widths and a run (i0,i1), return:
      (overall_length_px, underhead_length_px)
    Head is detected as an end segment where width > head_ratio * shank_width.
    """
    i0, i1 = run
    seg = widths[i0:i1]
    run_len = i1 - i0

    # Shank width estimate from central band (avoid both ends)
    a = i0 + int(0.35 * run_len)
    b = i0 + int(0.65 * run_len)
    core = widths[a:b]
    core = core[core > 0]
    if len(core) < 10:
        core = seg[seg > 0]
    shank = float(np.median(core)) if len(core) else float(np.median(seg))

    thresh = head_ratio * shank

    # Find head segment length from top
    top = 0
    while (i0 + top) < i1 and widths[i0 + top] > thresh:
        top += 1

    # Find head segment length from bottom
    bot = 0
    while (i1 - 1 - bot) >= i0 and widths[i1 - 1 - bot] > thresh:
        bot += 1

    # Require a minimum to count as head; else treat as no-head-detected
    min_head = max(6, int(0.08 * run_len))
    top = top if top >= min_head else 0
    bot = bot if bot >= min_head else 0

    # Choose larger head segment as the head side
    head_px = float(max(top, bot))
    overall_px = float(run_len)
    underhead_px = float(max(0.0, overall_px - head_px))
    return overall_px, underhead_px


def guess_metric_size(diameter_mm: float) -> str:
    sizes = [3.0, 4.0, 5.0, 6.0]
    labels = ["M3", "M4", "M5", "M6"]
    if diameter_mm <= 0:
        return "unknown"
    i = int(np.argmin([abs(diameter_mm - s) for s in sizes]))
    return labels[i] if abs(diameter_mm - sizes[i]) <= 1.0 else "unknown"


def measure_screws(
    bgr: np.ndarray,
    grid_mm: float,
    bright_thresh: int,
    bh_ksize: int,
    bh_thresh: int,
    min_ar: float,
    local_k: int,
) -> Tuple[List[ScrewMeasurement], np.ndarray, float]:
    roi = auto_crop_camera_region(bgr)
    vis = roi.copy()
    gray0 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Grid centers for scale + local correction
    _, centers0 = detect_grid_squares(gray0, bright_thresh=bright_thresh)
    pitch_px_global = estimate_pitch_px_nn(centers0) if centers0 else None
    if pitch_px_global is None:
        pitch_px_global = 50.0
    px_per_mm_global = pitch_px_global / grid_mm

    # Crop to grid bounds to reduce vignette artifacts
    gray = gray0
    ox = oy = 0
    centers = centers0
    if len(centers0) >= 20:
        pad_px = int(0.8 * pitch_px_global)
        gray, (ox, oy) = crop_to_grid(gray0, centers0, pad_px=pad_px)
        # shift centers into cropped coordinates
        centers = [(x - ox, y - oy) for (x, y) in centers0]

    # Precompute per-center nearest-neighbor distance for local pitch estimation
    pts = np.array(centers, dtype=np.float32) if len(centers) else None
    nn_dist = None
    if pts is not None and len(pts) >= 20:
        nn_dist = np.zeros(len(pts), dtype=np.float32)
        for i in range(len(pts)):
            d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
            d = d[d > 1e-3]
            nn_dist[i] = float(d.min()) if len(d) else float(pitch_px_global)

    # Candidate detection (black-hat)
    k = max(9, int(bh_ksize) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)
    bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    _, cand = cv2.threshold(bh, bh_thresh, 255, cv2.THRESH_BINARY)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dynamic min area
    min_area = int(max(200, (px_per_mm_global ** 2) * 5.0))

    raw = []
    H, W = gray.shape
    for c in cnts:
        area = int(cv2.contourArea(c))
        if area < min_area:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), _ = rect
        L = max(rw, rh)
        Wid = min(rw, rh)
        if Wid < 6 or L < 35:
            continue
        ar = L / (Wid + 1e-6)
        if ar < min_ar:
            continue
        # ignore near-border
        if cx < 5 or cy < 5 or cx > (W - 5) or cy > (H - 5):
            continue
        raw.append((area, rect))

    raw.sort(key=lambda t: t[0], reverse=True)

    # De-dup by center distance
    kept = []
    kept_centers = []
    for area, rect in raw:
        (cx, cy), _, _ = rect
        if any((cx - kx) ** 2 + (cy - ky) ** 2 < (0.6 * pitch_px_global) ** 2 for kx, ky in kept_centers):
            continue
        kept.append((area, rect))
        kept_centers.append((cx, cy))

    def local_px_per_mm(cx: float, cy: float) -> float:
        if pts is None or nn_dist is None or len(pts) < 20:
            return px_per_mm_global
        d2 = ((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        idx = np.argsort(d2)[:max(10, local_k)]
        lp = float(np.median(nn_dist[idx]))
        if lp < 10 or lp > 300:
            return px_per_mm_global
        return lp / grid_mm

    measurements: List[ScrewMeasurement] = []
    sid = 1

    for _, rect in kept:
        (cx, cy), _, ang = rect

        # local scale (perspective compensation)
        ppm = local_px_per_mm(cx, cy)

        patch = rotate_patch(gray, rect, pad=55)
        widths, _ = width_profile_contiguous_run(patch, search=14)

        min_w_px = max(5, int(0.6 * ppm))  # ~0.6mm
        run = longest_good_run(widths, min_w_px=min_w_px)
        if run is None:
            continue

        overall_px, underhead_px = estimate_underhead_length_px(widths, run, head_ratio=1.35)

        # shank diameter from lower quantile within run (avoid head)
        i0, i1 = run
        seg = widths[i0:i1]
        seg = seg[seg >= float(min_w_px)]
        if len(seg) < 10:
            continue
        cut = np.percentile(seg, 40)
        shank = seg[seg <= cut]
        dia_px = float(np.median(shank)) if len(shank) else float(np.median(seg))

        length_overall_mm = overall_px / ppm
        length_underhead_mm = underhead_px / ppm
        dia_mm = dia_px / ppm

        # plausibility filters
        if not (3.0 <= length_overall_mm <= 80.0):
            continue
        if not (1.5 <= dia_mm <= 8.0):
            continue

        # shift back to original ROI coordinates
        cx0 = cx + ox
        cy0 = cy + oy

        m_guess = guess_metric_size(dia_mm)

        measurements.append(
            ScrewMeasurement(
                id=sid,
                center_px=(float(cx0), float(cy0)),
                length_overall_mm=float(length_overall_mm),
                length_underhead_mm=float(length_underhead_mm),
                diameter_mm=float(dia_mm),
                m_guess=m_guess,
            )
        )
        sid += 1

    # annotate
    for m in measurements:
        cx, cy = m.center_px
        label = (
            f"{m.id}: {m.m_guess} "
            f"L={m.length_underhead_mm:.1f}mm "
            f"(ovr {m.length_overall_mm:.1f}) "
            f"D={m.diameter_mm:.1f}mm"
        )
        cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 0), 2)
        cv2.putText(vis, label, (int(cx) - 140, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)

    return measurements, vis, px_per_mm_global


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Local path or http(s) URL")
    ap.add_argument("--grid-mm", type=float, default=5.0)
    ap.add_argument("--bright-thresh", type=int, default=190)
    ap.add_argument("--bh-ksize", type=int, default=21)
    ap.add_argument("--bh-thresh", type=int, default=35)
    ap.add_argument("--min-ar", type=float, default=1.25)
    ap.add_argument("--local-k", type=int, default=40, help="How many nearby grid points to use for local scale")
    ap.add_argument("--out", default="annotated_v4.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    bgr = load_image(args.image)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")

    meas, ann, px_per_mm_global = measure_screws(
        bgr=bgr,
        grid_mm=args.grid_mm,
        bright_thresh=args.bright_thresh,
        bh_ksize=args.bh_ksize,
        bh_thresh=args.bh_thresh,
        min_ar=args.min_ar,
        local_k=args.local_k,
    )

    cv2.imwrite(args.out, ann)

    out_json = {
        "px_per_mm_global": round(px_per_mm_global, 6),
        "grid_mm": args.grid_mm,
        "count": len(meas),
        "screws": [
            {
                "id": m.id,
                "m_guess": m.m_guess,
                "diameter_mm": round(m.diameter_mm, 3),
                "length_underhead_mm": round(m.length_underhead_mm, 3),
                "length_overall_mm": round(m.length_overall_mm, 3),
                "center_px": [round(m.center_px[0], 1), round(m.center_px[1], 1)],
            }
            for m in meas
        ],
        "annotated_image": os.path.abspath(args.out),
    }

    print(json.dumps(out_json, indent=2))

    if args.show:
        cv2.imshow("annotated", ann)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
