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
    length_mm: float
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
    """Crop UI screenshots to largest dark-ish camera pane; if already a raw frame, keep as-is."""
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


def detect_grid_squares(gray: np.ndarray, bright_thresh: int):
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(g, bright_thresh, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares, centers = [], []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 250 or area > 8000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8:
            continue
        ar = w / float(h)
        if ar < 0.7 or ar > 1.35:
            continue
        squares.append(c)
        centers.append((x + w / 2.0, y + h / 2.0))
    return squares, centers


def estimate_pitch_px_nn(centers: List[Tuple[float, float]]) -> Optional[float]:
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


def crop_to_grid(gray: np.ndarray, centers: List[Tuple[float, float]], pad_px: int):
    pts = np.array(centers, dtype=np.float32)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    x0 = int(clamp(xmin - pad_px, 0, gray.shape[1] - 1))
    y0 = int(clamp(ymin - pad_px, 0, gray.shape[0] - 1))
    x1 = int(clamp(xmax + pad_px, 1, gray.shape[1]))
    y1 = int(clamp(ymax + pad_px, 1, gray.shape[0]))
    return gray[y0:y1, x0:x1], (x0, y0)


def rotate_patch_mask(mask: np.ndarray, rect, pad: int = 60) -> np.ndarray:
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
    rot = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)

    bw = int(short_side + 2 * pad)
    bh = int(long_side + 2 * pad)
    x0 = int(cx - bw / 2)
    y0 = int(cy - bh / 2)
    x0 = clamp(x0, 0, mask.shape[1] - 1)
    y0 = clamp(y0, 0, mask.shape[0] - 1)
    x1 = clamp(x0 + bw, 1, mask.shape[1])
    y1 = clamp(y0 + bh, 1, mask.shape[0])

    patch = rot[y0:y1, x0:x1]
    if patch.shape[1] > patch.shape[0]:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    return patch


def mask_widths(mask_patch: np.ndarray) -> np.ndarray:
    h, w = mask_patch.shape
    widths = np.zeros(h, dtype=np.float32)
    for r in range(h):
        row = mask_patch[r]
        xs = np.where(row > 0)[0]
        if xs.size == 0:
            continue
        # largest contiguous segment width
        dif = np.diff(xs)
        breaks = np.where(dif > 1)[0]
        start = 0
        best = 0
        for b in np.r_[breaks, xs.size - 1]:
            seg = xs[start:b + 1]
            best = max(best, seg.size)
            start = b + 1
        widths[r] = best
    return widths


def longest_run(widths: np.ndarray, min_w_px: int) -> Optional[Tuple[int, int]]:
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


def guess_metric_size(diameter_mm: float) -> str:
    sizes = [3.0, 4.0, 5.0, 6.0]
    labels = ["M3", "M4", "M5", "M6"]
    if diameter_mm <= 0:
        return "unknown"
    i = int(np.argmin([abs(diameter_mm - s) for s in sizes]))
    return labels[i] if abs(diameter_mm - sizes[i]) <= 1.0 else "unknown"


def detect_candidates_blackhat(gray: np.ndarray, bh_ksize: int, bh_thresh: int, close_ks: int, close_it: int):
    k = max(9, int(bh_ksize) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)
    bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    _, cand = cv2.threshold(bh, int(bh_thresh), 255, cv2.THRESH_BINARY)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((int(close_ks), int(close_ks)), np.uint8), iterations=int(close_it))
    return cand


def detect_candidates_dark(gray: np.ndarray, squares_mask_dil: np.ndarray, perc: float, open_it: int, close_ks: int, close_it: int):
    g = np.minimum(gray, 220)
    t = float(np.percentile(g, perc))
    dark = (g < t).astype(np.uint8) * 255
    dark = cv2.bitwise_and(dark, cv2.bitwise_not(squares_mask_dil))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=int(open_it))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((int(close_ks), int(close_ks)), np.uint8), iterations=int(close_it))
    return dark


def measure_screws(
    bgr: np.ndarray,
    grid_mm: float,
    bright_thresh: int,
    method: str,
    bh_ksize: int,
    bh_thresh: int,
    bh_close_ks: int,
    bh_close_it: int,
    dark_perc: float,
    dark_open_it: int,
    dark_close_ks: int,
    dark_close_it: int,
    dark_dilate: int,
    min_ar: float,
):
    roi = auto_crop_camera_region(bgr)
    vis = roi.copy()
    gray0 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    squares, centers = detect_grid_squares(gray0, bright_thresh=bright_thresh)
    pitch_px = estimate_pitch_px_nn(centers) if centers else None
    if pitch_px is None:
        pitch_px = 50.0
    px_per_mm = pitch_px / grid_mm

    # crop to grid region to reduce vignette
    gray = gray0
    ox = oy = 0
    if len(centers) >= 20:
        pad_px = int(0.8 * pitch_px)
        gray, (ox, oy) = crop_to_grid(gray0, centers, pad_px=pad_px)

    # build squares mask (same crop) to remove hole regions
    sq_mask_full = np.zeros_like(gray0, dtype=np.uint8)
    for c in squares:
        cv2.drawContours(sq_mask_full, [c], -1, 255, thickness=cv2.FILLED)
    sq_mask = sq_mask_full[oy:oy + gray.shape[0], ox:ox + gray.shape[1]]
    sq_mask_dil = cv2.dilate(sq_mask, np.ones((9, 9), np.uint8), iterations=1)

    # candidates
    cand = None
    tried = []
    if method in ("auto", "blackhat"):
        cand = detect_candidates_blackhat(gray, bh_ksize, bh_thresh, bh_close_ks, bh_close_it)
        tried.append("blackhat")
    # quick reject if essentially blank or one huge blob
    cnts = []
    if cand is not None:
        cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if method in ("auto", "dark") and (method == "dark" or len(cnts) == 0):
        cand = detect_candidates_dark(gray, sq_mask_dil, dark_perc, dark_open_it, dark_close_ks, dark_close_it)
        tried.append("dark")
        cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # filter contours
    min_area = int(max(200, (px_per_mm ** 2) * 4.0))
    H, W = gray.shape
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), _ = rect
        L = max(rw, rh)
        Wid = min(rw, rh)
        if Wid < 6 or L < 30:
            continue
        ar = L / (Wid + 1e-6)
        if ar < min_ar:
            continue
        # reject huge blobs spanning most of ROI
        if rw > 0.85 * W and rh > 0.85 * H:
            continue
        # reject border/vignette junk
        if cx < 8 or cy < 8 or cx > (W - 8) or cy > (H - 8):
            continue
        candidates.append((area, rect, c))

    candidates.sort(key=lambda t: t[0], reverse=True)

    # measure from contour mask (robust against grid texture)
    meas: List[ScrewMeasurement] = []
    sid = 1

    base_mask = np.zeros_like(gray, dtype=np.uint8)

    for _, rect, contour in candidates:
        base_mask[:] = 0
        cv2.drawContours(base_mask, [contour], -1, 255, thickness=cv2.FILLED)

        if "dark" in tried:
            # dark method tends to under-segment edges; dilate a bit to recover silhouette
            if dark_dilate > 0:
                base_mask = cv2.dilate(base_mask, np.ones((3, 3), np.uint8), iterations=int(dark_dilate))

        mp = rotate_patch_mask(base_mask, rect, pad=65)
        widths = mask_widths(mp)

        min_w_px = max(5, int(0.6 * px_per_mm))  # ~0.6mm
        run = longest_run(widths, min_w_px=min_w_px)
        if run is None:
            continue

        i0, i1 = run
        length_px = float(i1 - i0)

        seg = widths[i0:i1]
        seg = seg[seg >= float(min_w_px)]
        if len(seg) < 10:
            continue

        # diameter estimate from central region to avoid head influence
        a = int(i0 + 0.35 * (i1 - i0))
        b = int(i0 + 0.65 * (i1 - i0))
        core = widths[a:b]
        core = core[core >= float(min_w_px)]
        use = core if len(core) >= 10 else seg

        # use lower quantile median to bias toward shank
        cut = np.percentile(use, 40)
        shank = use[use <= cut]
        dia_px = float(np.median(shank)) if len(shank) else float(np.median(use))

        length_mm = length_px / px_per_mm
        dia_mm = dia_px / px_per_mm

        # plausibility
        if not (3.0 <= length_mm <= 100.0):
            continue
        if not (1.5 <= dia_mm <= 8.0):
            continue

        (cx, cy), _, _ = rect
        cx0, cy0 = float(cx + ox), float(cy + oy)

        m_guess = guess_metric_size(dia_mm)
        meas.append(ScrewMeasurement(sid, (cx0, cy0), length_mm, dia_mm, m_guess))
        sid += 1

    # annotate
    for m in meas:
        cx, cy = m.center_px
        label = f"{m.id}: {m.m_guess} L={m.length_mm:.1f}mm D={m.diameter_mm:.1f}mm"
        cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 0), 2)
        cv2.putText(vis, label, (int(cx) - 140, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    out_json = {
        "px_per_mm_global": round(px_per_mm, 6),
        "grid_mm": grid_mm,
        "method_used": tried[-1] if tried else method,
        "count": len(meas),
        "screws": [
            {
                "id": m.id,
                "m_guess": m.m_guess,
                "diameter_mm": round(m.diameter_mm, 3),
                "length_mm": round(m.length_mm, 3),
                "center_px": [round(m.center_px[0], 1), round(m.center_px[1], 1)],
            }
            for m in meas
        ],
    }

    return out_json, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Local path or http(s) URL")
    ap.add_argument("--grid-mm", type=float, default=5.0)
    ap.add_argument("--bright-thresh", type=int, default=190)

    ap.add_argument("--method", choices=["auto", "blackhat", "dark"], default="auto")

    # blackhat params
    ap.add_argument("--bh-ksize", type=int, default=21)
    ap.add_argument("--bh-thresh", type=int, default=35)
    ap.add_argument("--bh-close-ks", type=int, default=11)
    ap.add_argument("--bh-close-it", type=int, default=2)

    # dark params
    ap.add_argument("--dark-perc", type=float, default=20.0, help="Percentile for dark threshold (higher = more pixels)")
    ap.add_argument("--dark-open-it", type=int, default=1)
    ap.add_argument("--dark-close-ks", type=int, default=13)
    ap.add_argument("--dark-close-it", type=int, default=2)
    ap.add_argument("--dark-dilate", type=int, default=3, help="Recover silhouette after dark threshold")

    ap.add_argument("--min-ar", type=float, default=1.20)

    ap.add_argument("--out", default="annotated_v5.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    bgr = load_image(args.image)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")

    out_json, ann = measure_screws(
        bgr=bgr,
        grid_mm=args.grid_mm,
        bright_thresh=args.bright_thresh,
        method=args.method,
        bh_ksize=args.bh_ksize,
        bh_thresh=args.bh_thresh,
        bh_close_ks=args.bh_close_ks,
        bh_close_it=args.bh_close_it,
        dark_perc=args.dark_perc,
        dark_open_it=args.dark_open_it,
        dark_close_ks=args.dark_close_ks,
        dark_close_it=args.dark_close_it,
        dark_dilate=args.dark_dilate,
        min_ar=args.min_ar,
    )

    cv2.imwrite(args.out, ann)
    out_json["annotated_image"] = os.path.abspath(args.out)
    print(json.dumps(out_json, indent=2))

    if args.show:
        cv2.imshow("annotated", ann)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
