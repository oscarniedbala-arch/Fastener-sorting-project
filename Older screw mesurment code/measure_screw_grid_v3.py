#!/usr/bin/env python3
import argparse
import json
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
    angle_deg: float


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def auto_crop_camera_region(bgr: np.ndarray) -> np.ndarray:
    """If input is a UI screenshot, crop to largest dark-ish rectangle (camera pane)."""
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


def detect_grid_squares(gray: np.ndarray, bright_thresh: int) -> Tuple[List[np.ndarray], List[Tuple[float, float]], np.ndarray]:
    """Detect bright square holes; return contours, centers, and filled mask."""
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

    mask = np.zeros_like(gray, dtype=np.uint8)
    if squares:
        cv2.drawContours(mask, squares, -1, 255, thickness=cv2.FILLED)
    return squares, centers, mask


def estimate_pitch_px_nn(centers: List[Tuple[float, float]]) -> Optional[float]:
    """Robust pitch estimate: median nearest-neighbor distance."""
    if len(centers) < 20:
        return None
    pts = np.array(centers, dtype=np.float32)
    dmins = []
    for i in range(len(pts)):
        di = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
        di = di[di > 1e-3]
        dmins.append(float(di.min()))
    pitch = float(np.median(dmins))
    if pitch < 10 or pitch > 300:
        return None
    return pitch


def crop_to_grid(gray: np.ndarray, centers: List[Tuple[float, float]], pad_px: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop image to bounding box of detected grid centers + padding."""
    pts = np.array(centers, dtype=np.float32)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    x0 = int(clamp(xmin - pad_px, 0, gray.shape[1] - 1))
    y0 = int(clamp(ymin - pad_px, 0, gray.shape[0] - 1))
    x1 = int(clamp(xmax + pad_px, 1, gray.shape[1]))
    y1 = int(clamp(ymax + pad_px, 1, gray.shape[0]))
    return gray[y0:y1, x0:x1], (x0, y0)


def rotate_patch(gray: np.ndarray, rect, pad: int = 40) -> np.ndarray:
    """Extract a rotated patch aligned with the screw major axis (vertical)."""
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
    # Ensure tall
    if patch.shape[1] > patch.shape[0]:
        patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    return patch


def measure_patch_run(patch: np.ndarray, px_per_mm: float, search: int = 14) -> Optional[Tuple[float, float]]:
    """
    Measure by per-row contiguous dark run near the screw axis.
    This avoids counting grid/shadow pixels elsewhere on the row.
    """
    if patch.size == 0:
        return None

    p = patch.copy().astype(np.uint8)
    # Clip very bright holes so Otsu isn't dominated by them
    p = np.minimum(p, 220)
    p = cv2.GaussianBlur(p, (3, 3), 0)

    # Otsu threshold (dark pixels < thr)
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

    # Keep rows that are “wide enough” to be screw, not noise
    min_w_px = max(5, int(0.6 * px_per_mm))  # ~0.6 mm
    good = widths >= min_w_px

    # Longest consecutive run of "good" rows = screw length in pixels
    best_len = 0
    best = (0, 0)
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

    if best_len < 20:
        return None

    i0, i1 = best
    length_px = float(best_len)

    # Diameter from central section; use lower quantile to avoid head
    a = i0 + int(0.12 * best_len)
    b = i1 - int(0.12 * best_len)
    core = widths[a:b]
    core = core[core >= min_w_px]
    if len(core) < 10:
        core = widths[i0:i1]
        core = core[core >= min_w_px]

    cut = np.percentile(core, 40)  # shank tends to be narrower than head
    shank = core[core <= cut]
    dia_px = float(np.median(shank)) if len(shank) else float(np.median(core))

    length_mm = length_px / px_per_mm
    dia_mm = dia_px / px_per_mm
    return float(length_mm), float(dia_mm)


def guess_metric_size(diameter_mm: float) -> str:
    sizes = [3.0, 4.0, 5.0, 6.0]
    labels = ["M3", "M4", "M5", "M6"]
    if diameter_mm <= 0:
        return "unknown"
    i = int(np.argmin([abs(diameter_mm - s) for s in sizes]))
    if abs(diameter_mm - sizes[i]) > 1.0:
        return "unknown"
    return labels[i]


def measure_screws(
    bgr: np.ndarray,
    grid_mm: float,
    bright_thresh: int,
    bh_ksize: int,
    bh_thresh: int,
    min_area: int,
    min_ar: float,
) -> Tuple[List[ScrewMeasurement], np.ndarray, float]:
    roi = auto_crop_camera_region(bgr)
    gray0 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    vis0 = roi.copy()

    # Grid detection and scale
    _, centers, _ = detect_grid_squares(gray0, bright_thresh=bright_thresh)
    pitch_px = estimate_pitch_px_nn(centers) if centers else None
    if pitch_px is None:
        pitch_px = 50.0  # fallback
    px_per_mm = pitch_px / grid_mm

    # Crop to grid bounds to remove vignette/border artifacts
    if len(centers) >= 20:
        pad_px = int(0.8 * pitch_px)
        gray, (ox, oy) = crop_to_grid(gray0, centers, pad_px=pad_px)
        vis = vis0[oy:oy + gray.shape[0], ox:ox + gray.shape[1]].copy()
    else:
        gray, vis = gray0, vis0
        ox, oy = 0, 0

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

    # Score candidates; then do a simple non-maximum suppression by center distance
    raw = []
    H, W = gray.shape
    for c in cnts:
        area = int(cv2.contourArea(c))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or (x + w) >= (W - 2) or (y + h) >= (H - 2):
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        L = max(rw, rh)
        Wid = min(rw, rh)
        if Wid < 6:
            continue
        ar = L / (Wid + 1e-6)
        if ar < min_ar or L < 35:
            continue

        raw.append((area, rect))

    raw.sort(key=lambda t: t[0], reverse=True)

    kept = []
    centers_kept = []
    for area, rect in raw:
        (cx, cy), (rw, rh), _ = rect
        too_close = False
        for (kx, ky) in centers_kept:
            if (cx - kx) ** 2 + (cy - ky) ** 2 < (0.6 * pitch_px) ** 2:
                too_close = True
                break
        if not too_close:
            kept.append((area, rect))
            centers_kept.append((cx, cy))

    measurements: List[ScrewMeasurement] = []
    sid = 1
    for area, rect in kept:
        patch = rotate_patch(gray, rect, pad=55)
        res = measure_patch_run(patch, px_per_mm, search=14)
        if res is None:
            continue
        length_mm, dia_mm = res

        # Plausibility filter (tune if you use very large screws)
        if not (3.0 <= length_mm <= 80.0):
            continue
        if not (1.5 <= dia_mm <= 8.0):
            continue

        (cx, cy), (_, _), ang = rect
        m_guess = guess_metric_size(dia_mm)

        measurements.append(
            ScrewMeasurement(
                id=sid,
                center_px=(float(cx + ox), float(cy + oy)),
                length_mm=float(length_mm),
                diameter_mm=float(dia_mm),
                m_guess=m_guess,
                angle_deg=float(ang),
            )
        )
        sid += 1

    # Annotate
    for m in measurements:
        cx, cy = m.center_px
        label = f"{m.id}: {m.m_guess} L={m.length_mm:.1f}mm D={m.diameter_mm:.1f}mm"
        cv2.circle(vis0, (int(cx), int(cy)), 6, (0, 255, 0), 2)
        cv2.putText(
            vis0,
            label,
            (int(cx) - 90, int(cy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return measurements, vis0, px_per_mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to capture.jpg / screenshot.png")
    ap.add_argument("--grid-mm", type=float, default=5.0)
    ap.add_argument("--bright-thresh", type=int, default=190)
    ap.add_argument("--bh-ksize", type=int, default=21)
    ap.add_argument("--bh-thresh", type=int, default=35)
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--min-ar", type=float, default=1.25)
    ap.add_argument("--out", default="annotated.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")

    meas, ann, px_per_mm = measure_screws(
        bgr=bgr,
        grid_mm=args.grid_mm,
        bright_thresh=args.bright_thresh,
        bh_ksize=args.bh_ksize,
        bh_thresh=args.bh_thresh,
        min_area=args.min_area,
        min_ar=args.min_ar,
    )

    cv2.imwrite(args.out, ann)

    out_json = {
        "px_per_mm": round(px_per_mm, 6),
        "grid_mm": args.grid_mm,
        "count": len(meas),
        "screws": [
            {
                "id": m.id,
                "length_mm": round(m.length_mm, 3),
                "diameter_mm": round(m.diameter_mm, 3),
                "m_guess": m.m_guess,
                "center_px": [round(m.center_px[0], 1), round(m.center_px[1], 1)],

            }
            for m in meas
        ],
        "annotated_image": args.out,
    }

    print(json.dumps(out_json, indent=2))

    if args.show:
        cv2.imshow("annotated", ann)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
