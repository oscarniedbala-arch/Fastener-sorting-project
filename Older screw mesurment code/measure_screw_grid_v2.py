#!/usr/bin/env python3
"""
measure_screw_grid_v2.py (fixed / cleaned)

CV-only screw measurement using a known-size grid background.

How it works:
  1) Detect the grid squares (adaptive threshold + shape filtering).
  2) Estimate px/mm from median square size (or override with --px-per-mm).
  3) Canny edges for objects, then suppress edges near grid squares.
  4) Morphological fill to get candidate blobs.
  5) Choose the most "screw-like" blob (elongated, solid, not huge).
  6) Rotate to horizontal and estimate:
       - total length (mm)
       - shaft diameter (mm)
       - under-head length (best effort)

Notes:
- --v-min/--v-max are used as Canny thresholds.
- --s-max is kept for backwards compatibility but is not used.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import requests


# -----------------------------
# Data model
# -----------------------------
@dataclass
class ScrewMeasurement:
    px_per_mm: float
    total_length_mm: float
    under_head_length_mm: Optional[float]
    shaft_diam_mm: float
    head_diam_mm: Optional[float]
    confidence: float


# -----------------------------
# IO
# -----------------------------
def fetch_image_bgr(url: str, timeout: int = 10) -> np.ndarray:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image from URL")
    return img


# -----------------------------
# Grid detection + px/mm
# -----------------------------
def _adaptive_square_candidates(gray: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, -5
    )
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return thr


def detect_grid_squares_mask(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      square_mask: uint8 {0,255}
      sizes_px: float32 array of detected square size estimate in pixels
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thr = _adaptive_square_candidates(gray)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    square_mask = np.zeros_like(thr)
    sizes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # Very broad limits (safe across VGA->128 crops)
        if area < 60 or area > 60000:
            continue

        ar = w / float(h + 1e-6)
        if ar < 0.75 or ar > 1.35:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        cv2.drawContours(square_mask, [c], -1, 255, thickness=-1)
        sizes.append((w + h) / 2.0)

    return square_mask, np.asarray(sizes, dtype=np.float32)


def estimate_px_per_mm_from_grid(bgr: np.ndarray, square_mm: float) -> Optional[float]:
    _, sizes = detect_grid_squares_mask(bgr)
    if sizes.size < 4:
        return None
    square_px = float(np.median(sizes))
    return square_px / float(square_mm)


# -----------------------------
# Segmentation (grid-suppressed edges)
# -----------------------------
def segment_screw_mask(
    bgr: np.ndarray,
    s_max: int = 0,          # kept for compatibility (unused)
    v_min: int = 40,         # Canny low threshold
    v_max: int = 140,        # Canny high threshold
    square_dilate: int = 19, # how aggressively to suppress grid edges
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      blob_mask   : uint8 {0,255}
      square_mask : uint8 {0,255}
      edges_clean : uint8 {0,255}
    """
    if square_dilate < 3:
        square_dilate = 3
    if square_dilate % 2 == 0:
        square_dilate += 1

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    square_mask, _ = detect_grid_squares_mask(bgr)

    edges = cv2.Canny(gray_blur, int(v_min), int(v_max))

    suppress = cv2.dilate(square_mask, np.ones((square_dilate, square_dilate), np.uint8), iterations=1)
    edges_clean = cv2.bitwise_and(edges, cv2.bitwise_not(suppress))

    blob = cv2.dilate(edges_clean, np.ones((5, 5), np.uint8), iterations=2)
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=2)

    return blob, square_mask, edges_clean


# -----------------------------
# Candidate selection
# -----------------------------
def best_screw_contour(
    mask: np.ndarray,
    min_area: float = 600.0,
    min_rot_ar: float = 1.4,
    min_solidity: float = 0.70,
    max_bbox_frac: float = 0.50,
) -> Optional[np.ndarray]:
    h, w = mask.shape[:2]
    frame_area = float(h * w)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = -1.0

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if float(bw * bh) > max_bbox_frac * frame_area:
            continue

        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), _ = rect
        if rw < 5 or rh < 5:
            continue

        rot_ar = max(rw, rh) / float(min(rw, rh) + 1e-6)
        if rot_ar < min_rot_ar:
            continue

        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull)) + 1e-6
        solidity = area / hull_area
        if solidity < min_solidity:
            continue

        score = area * min(rot_ar, 8.0) * solidity
        if score > best_score:
            best_score = score
            best = c

    return best


# -----------------------------
# Geometry helpers
# -----------------------------
def rotate_image_and_mask(bgr: np.ndarray, mask: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    rbgr = cv2.warpAffine(bgr, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rmask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rbgr, rmask


def width_profile(bin_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(bin_mask > 0)
    if xs.size == 0:
        return np.array([], dtype=np.float32)
    x0, x1 = int(xs.min()), int(xs.max())
    prof = np.zeros(x1 - x0 + 1, dtype=np.float32)
    for i, x in enumerate(range(x0, x1 + 1)):
        prof[i] = float(np.count_nonzero(bin_mask[:, x]))
    return prof


def smooth_1d(a: np.ndarray, k: int = 11) -> np.ndarray:
    if a.size < k:
        return a
    k = max(3, int(k) | 1)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(a, kernel, mode="same")


def classify_gauge(d_mm: float) -> str:
    # Calibrate these thresholds using your known screws.
    if d_mm < 4.5:
        return "M4-ish"
    if d_mm < 5.5:
        return "M5-ish"
    if d_mm < 6.6:
        return "M6-ish"
    return "Unknown"


# -----------------------------
# Measurement
# -----------------------------
def measure_screw(
    bgr: np.ndarray,
    square_mm: float = 5.0,
    px_per_mm_override: Optional[float] = None,
    s_max: int = 0,
    v_min: int = 40,
    v_max: int = 140,
) -> Tuple[ScrewMeasurement, np.ndarray, dict]:
    if px_per_mm_override is not None:
        px_per_mm = float(px_per_mm_override)
    else:
        px_per_mm = estimate_px_per_mm_from_grid(bgr, square_mm=square_mm)
        if px_per_mm is None:
            raise RuntimeError(
                "Could not estimate px/mm from grid. Ensure the grid is visible and well lit, "
                "or pass --px-per-mm to override."
            )

    blob, squares, edges_clean = segment_screw_mask(bgr, s_max=s_max, v_min=v_min, v_max=v_max)
    mask_meas = cv2.erode(blob, np.ones((3, 3), np.uint8), iterations=1)

    c = best_screw_contour(mask_meas)
    if c is None:
        dbg = {"mask_squares": squares, "edges_clean": edges_clean, "mask_screw": mask_meas}
        raise RuntimeError("No screw-like contour detected (tune --v-min/--v-max; reduce glare; keep hands out of frame).")

    rect = cv2.minAreaRect(c)
    (_, _), (rw, rh), angle = rect

    # Rotate to make long axis horizontal
    rot = angle if rw < rh else angle + 90.0
    _, rmask = rotate_image_and_mask(bgr, mask_meas, rot)

    ys, xs = np.where(rmask > 0)
    if xs.size == 0:
        raise RuntimeError("Internal error: rotated mask is empty.")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    roi = rmask[y0:y1 + 1, x0:x1 + 1]
    roi = (roi > 0).astype(np.uint8) * 255

    prof = width_profile(roi)
    if prof.size < 40:
        raise RuntimeError("Screw too small in frame; move camera closer or increase resolution.")
    prof = smooth_1d(prof, k=11)

    nz = prof[prof > 0]
    if nz.size < 20:
        raise RuntimeError("Profile extraction failed (mask too sparse).")

    # Total length: count columns above a small threshold
    shaft_guess = float(np.percentile(nz, 20))
    valid = prof > max(3.0, 0.25 * shaft_guess)
    idx = np.where(valid)[0]
    total_len_px = float(idx.max() - idx.min()) if idx.size else float(prof.size - 1)

    # Shaft diameter: mid-region percentile to reduce head influence
    n = prof.size
    mid = prof[int(0.25 * n): int(0.75 * n)]
    mid_nz = mid[mid > 0]
    if mid_nz.size < 10:
        raise RuntimeError("Profile mid-region too sparse.")
    shaft_px = float(np.percentile(mid_nz, 35))

    head_px = float(np.percentile(nz, 95))

    # Under-head length (best effort)
    left_mean = float(np.mean(prof[:max(8, n // 12)]))
    right_mean = float(np.mean(prof[-max(8, n // 12):]))
    head_on_left = left_mean > right_mean

    step_thresh = shaft_px * 1.35
    thick = np.where(prof > step_thresh)[0]
    under_head_px = None
    if thick.size > 0:
        if head_on_left:
            cand = thick[thick < n // 2]
            if cand.size > 0:
                boundary = int(np.max(cand))
                under_head_px = float((idx.max() if idx.size else (n - 1)) - boundary)
        else:
            cand = thick[thick > n // 2]
            if cand.size > 0:
                boundary = int(np.min(cand))
                under_head_px = float(boundary - (idx.min() if idx.size else 0))

    total_length_mm = total_len_px / px_per_mm
    shaft_diam_mm = shaft_px / px_per_mm
    head_diam_mm = (head_px / px_per_mm) if head_px > (shaft_px * 1.20) else None
    under_head_mm = (under_head_px / px_per_mm) if under_head_px is not None else None

    rot_ar = max(rw, rh) / float(min(rw, rh) + 1e-6)
    contrast = (head_px - shaft_px) / max(shaft_px, 1.0)
    confidence = float(np.clip(0.30 + 0.30 * np.clip(rot_ar / 4.0, 0.0, 1.0) + 0.40 * np.clip(contrast, 0.0, 1.0), 0.0, 1.0))

    # Annotate
    ann = bgr.copy()
    cv2.drawContours(ann, [c], -1, (0, 0, 255), 2)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(ann, [box], 0, (255, 0, 0), 2)

    gauge = classify_gauge(shaft_diam_mm)
    t1 = f"px/mm={px_per_mm:.2f}"
    t2 = f"L_total={total_length_mm:.1f}mm  D_shaft~{shaft_diam_mm:.2f}mm ({gauge})"
    t3 = f"L_under_head={under_head_mm:.1f}mm" if under_head_mm else "L_under_head=(n/a)"

    cv2.putText(ann, t1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(ann, t1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(ann, t2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(ann, t2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(ann, t3, (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(ann, t3, (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    dbg = {"mask_squares": squares, "edges_clean": edges_clean, "mask_screw": mask_meas}

    meas = ScrewMeasurement(
        px_per_mm=px_per_mm,
        total_length_mm=total_length_mm,
        under_head_length_mm=under_head_mm,
        shaft_diam_mm=shaft_diam_mm,
        head_diam_mm=head_diam_mm,
        confidence=confidence,
    )
    return meas, ann, dbg


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Measure screw length/diameter using a printed grid background.")
    ap.add_argument("--url", help="ESP32 image URL (e.g. http://192.168.0.202/capture.jpg)")
    ap.add_argument("--image", help="Local image file path")

    ap.add_argument("--square-mm", type=float, default=5.0, help="Grid square size in mm (default: 5.0)")
    ap.add_argument("--px-per-mm", type=float, default=None, help="Override px/mm (skips grid detection)")

    # Backward compatible flags
    ap.add_argument("--s-max", type=int, default=0, help="Unused (kept for compatibility).")
    ap.add_argument("--v-min", type=int, default=40, help="Canny low threshold (default: 40)")
    ap.add_argument("--v-max", type=int, default=140, help="Canny high threshold (default: 140)")

    ap.add_argument("--save", help="Save annotated image to this path")
    ap.add_argument("--show", action="store_true", help="Show annotated window")
    ap.add_argument("--debug", action="store_true", help="Show debug windows")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.url and not args.image:
        print("Provide --url or --image", file=sys.stderr)
        raise SystemExit(2)

    if args.url:
        bgr = fetch_image_bgr(args.url)
    else:
        bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to read image file")

    meas, ann, dbg = measure_screw(
        bgr,
        square_mm=args.square_mm,
        px_per_mm_override=args.px_per_mm,
        s_max=args.s_max,
        v_min=args.v_min,
        v_max=args.v_max,
    )

    print(f"px_per_mm:            {meas.px_per_mm:.3f}")
    print(f"total_length_mm:      {meas.total_length_mm:.2f}")
    print(f"under_head_length_mm: {meas.under_head_length_mm if meas.under_head_length_mm is not None else '(n/a)'}")
    print(f"shaft_diam_mm:        {meas.shaft_diam_mm:.3f}")
    print(f"head_diam_mm:         {meas.head_diam_mm if meas.head_diam_mm is not None else '(n/a)'}")
    print(f"confidence:           {meas.confidence:.2f}")

    if args.save:
        cv2.imwrite(args.save, ann)
        print(f"Saved annotated: {args.save}")

    if args.debug:
        cv2.imshow("mask_squares", dbg["mask_squares"])
        cv2.imshow("edges_clean", dbg["edges_clean"])
        cv2.imshow("mask_screw", dbg["mask_screw"])

    if args.show:
        cv2.imshow("annotated", ann)

    if args.show or args.debug:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
