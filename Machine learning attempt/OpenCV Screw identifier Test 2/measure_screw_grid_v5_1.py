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

    squares, centers, sizes = [], [], []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 250 or area > 9000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8:
            continue
        ar = w / float(h)
        if ar < 0.7 or ar > 1.35:
            continue
        squares.append(c)
        centers.append((x + w / 2.0, y + h / 2.0))
        sizes.append((w, h))
    return squares, centers, sizes


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
        xs = np.where(mask_patch[r] > 0)[0]
        if xs.size == 0:
            continue
        widths[r] = float(xs.max() - xs.min() + 1)
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
    return best if best_len >= 25 else None


def guess_metric_size(diameter_mm: float) -> str:
    sizes = [3.0, 4.0, 5.0, 6.0]
    labels = ["M3", "M4", "M5", "M6"]
    if diameter_mm <= 0:
        return "unknown"
    i = int(np.argmin([abs(diameter_mm - s) for s in sizes]))
    return labels[i] if abs(diameter_mm - sizes[i]) <= 1.2 else "unknown"


def measure(bgr: np.ndarray, square_mm: float, bright_thresh: int,
            dark_perc: float, min_ar: float, min_length_mm: float,
            silhouette_dilate: int):
    roi = auto_crop_camera_region(bgr)
    vis = roi.copy()
    gray0 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    squares, centers, sizes = detect_grid_squares(gray0, bright_thresh=bright_thresh)
    if len(centers) < 20:
        return {"error": "Could not detect enough grid squares for calibration."}, vis

    pitch_px = estimate_pitch_px_nn(centers) or 50.0
    hole_px = float(np.median([min(w, h) for (w, h) in sizes]))
    px_per_mm = hole_px / float(square_mm)  # << IMPORTANT: scale from hole size

    # crop to grid region
    pad_px = int(0.9 * pitch_px)
    gray, (ox, oy) = crop_to_grid(gray0, centers, pad_px=pad_px)

    # squares mask in cropped coords
    sq_mask_full = np.zeros_like(gray0, dtype=np.uint8)
    for c in squares:
        cv2.drawContours(sq_mask_full, [c], -1, 255, thickness=cv2.FILLED)
    sq_mask = sq_mask_full[oy:oy + gray.shape[0], ox:ox + gray.shape[1]]
    sq_mask = cv2.dilate(sq_mask, np.ones((9, 9), np.uint8), iterations=1)

    # inpaint away the bright holes so the grid stops dominating segmentation
    inpaint = cv2.inpaint(gray, sq_mask, 3, cv2.INPAINT_TELEA)

    # soften remaining periodic texture
    blur = cv2.GaussianBlur(inpaint, (0, 0), sigmaX=max(3.0, pitch_px / 6.0))
    # percentile threshold for dark objects
    t = float(np.percentile(blur, dark_perc))
    mask = (blur < t).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = inpaint.shape
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (px_per_mm ** 2) * 30:  # removes small junk
            continue
        x, y, wbb, hbb = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or x + wbb >= W - 3 or y + hbb >= H - 3:
            continue

        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), _ = rect
        L = max(rw, rh)
        Wid = min(rw, rh)
        ar = L / (Wid + 1e-6)
        if ar < min_ar:
            continue

        length_mm_est = float(L / px_per_mm)
        if length_mm_est < min_length_mm:
            continue

        candidates.append((area, rect, c))

    candidates.sort(key=lambda t: t[0], reverse=True)

    meas: List[ScrewMeasurement] = []
    sid = 1

    for _, rect, contour in candidates[:10]:
        # filled silhouette for measurement
        sil = np.zeros_like(inpaint, dtype=np.uint8)
        cv2.drawContours(sil, [contour], -1, 255, thickness=cv2.FILLED)
        if silhouette_dilate > 0:
            sil = cv2.dilate(sil, np.ones((3, 3), np.uint8), iterations=silhouette_dilate)

        patch = rotate_patch_mask(sil, rect, pad=70)
        widths = mask_widths(patch)

        min_w_px = max(6, int(0.9 * px_per_mm))  # ~0.9mm
        run = longest_run(widths, min_w_px=min_w_px)
        if run is None:
            continue
        i0, i1 = run
        length_px = float(i1 - i0)

        seg = widths[i0:i1]
        seg = seg[seg >= float(min_w_px)]
        if len(seg) < 10:
            continue

        # shank diameter: use lower-quantile median to avoid head region
        cut = np.percentile(seg, 45)
        shank = seg[seg <= cut]
        dia_px = float(np.median(shank)) if len(shank) else float(np.median(seg))

        length_mm = float(length_px / px_per_mm)
        dia_mm = float(dia_px / px_per_mm)

        (cx, cy), _, _ = rect
        cx0, cy0 = float(cx + ox), float(cy + oy)

        m_guess = guess_metric_size(dia_mm)
        meas.append(ScrewMeasurement(sid, (cx0, cy0), length_mm, dia_mm, m_guess))
        sid += 1

    for m in meas:
        cx, cy = m.center_px
        label = f"{m.id}: {m.m_guess} L={m.length_mm:.1f}mm D={m.diameter_mm:.1f}mm"
        cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 0), 2)
        cv2.putText(vis, label, (int(cx) - 160, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    out = {
        "square_mm": square_mm,
        "hole_px_median": round(hole_px, 3),
        "pitch_px_median": round(float(pitch_px), 3),
        "px_per_mm": round(float(px_per_mm), 6),
        "count": len(meas),
        "screws": [
            {
                "id": m.id,
                "m_guess": m.m_guess,
                "diameter_mm": round(m.diameter_mm, 3),
                "length_mm": round(m.length_mm, 3),
                "center_px": [round(m.center_px[0], 1), round(m.center_px[1], 1)],
            } for m in meas
        ],
    }
    return out, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--square-mm", type=float, default=5.0, help="Hole side length in mm (your 5x5mm squares)")
    ap.add_argument("--bright-thresh", type=int, default=190)
    ap.add_argument("--dark-perc", type=float, default=25.0, help="Percentile threshold for dark segmentation")
    ap.add_argument("--min-ar", type=float, default=1.5)
    ap.add_argument("--min-length-mm", type=float, default=12.0, help="Reject tiny false positives")
    ap.add_argument("--silhouette-dilate", type=int, default=2, help="Recover edges if mask is too skinny")
    ap.add_argument("--out", default="annotated_v5_1.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    bgr = load_image(args.image)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")

    out_json, ann = measure(
        bgr=bgr,
        square_mm=args.square_mm,
        bright_thresh=args.bright_thresh,
        dark_perc=args.dark_perc,
        min_ar=args.min_ar,
        min_length_mm=args.min_length_mm,
        silhouette_dilate=args.silhouette_dilate,
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
