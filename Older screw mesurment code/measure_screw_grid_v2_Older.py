import argparse
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import requests


@dataclass
class ScrewMeasurement:
    px_per_mm: float
    total_length_mm: float
    under_head_length_mm: float | None
    shaft_diam_mm: float
    head_diam_mm: float | None
    confidence: float


def fetch_image(url: str, timeout: int = 10) -> np.ndarray:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode JPEG from URL")
    return img


def estimate_px_per_mm_from_squares(bgr: np.ndarray, square_mm: float = 5.0) -> float | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold finds bright squares despite uneven lighting
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, -5
    )

    # Clean
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sizes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 60 or area > 40000:
            continue
        ar = w / float(h + 1e-6)
        if ar < 0.75 or ar > 1.35:
            continue

        # prefer "boxy" shapes
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        sizes.append((w + h) / 2.0)

    if not sizes:
        return None

    square_px = float(np.median(sizes))
    return square_px / square_mm



def segment_screw_mask(
    bgr: np.ndarray,
    s_max: int = 0,        # unused, kept for backward compatibility
    v_min: int = 40,       # Canny low threshold
    v_max: int = 140,      # Canny high threshold
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      screw_mask  : filled mask for screw candidate regions
      square_only : mask of grid squares
      edges_clean : edges after grid suppression (debug)
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Squares via adaptive threshold (robust to lighting)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, -5
    )
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    square_only = np.zeros_like(thr)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 60 or area > 40000:
            continue
        ar = w / float(h + 1e-6)
        if ar < 0.75 or ar > 1.35:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) < 4 or len(approx) > 6:
            continue
        cv2.drawContours(square_only, [c], -1, 255, thickness=-1)

    # --- Edges for objects
    edges = cv2.Canny(gray, v_min, v_max)

    # Suppress edges near squares (kills grid pattern)
    suppress = cv2.dilate(square_only, np.ones((19, 19), np.uint8), iterations=1)
    edges_clean = cv2.bitwise_and(edges, cv2.bitwise_not(suppress))

    # Build filled blobs from remaining edges
    blob = cv2.dilate(edges_clean, np.ones((5, 5), np.uint8), iterations=2)
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8), iterations=2)

    return blob, square_only, edges_clean



def best_screw_contour(mask: np.ndarray, min_area: float = 600.0, min_rot_ar: float = 1.4, min_solidity: float = 0.70) -> np.ndarray | None:
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

        # Reject huge blobs (background bands)
        if (bw * bh) > 0.50 * frame_area:
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


def rotate_image_and_mask(bgr: np.ndarray, mask: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
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


def width_profile(bin_mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    ys, xs = np.where(bin_mask > 0)
    if len(xs) == 0:
        return np.array([]), 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    prof = np.zeros(x1 - x0 + 1, dtype=np.float32)
    for i, x in enumerate(range(x0, x1 + 1)):
        prof[i] = float(np.count_nonzero(bin_mask[:, x]))
    return prof, x0, x1


def smooth_1d(a: np.ndarray, k: int = 11) -> np.ndarray:
    if len(a) < k:
        return a
    k = max(3, k | 1)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(a, kernel, mode="same")


def classify_gauge(d_mm: float) -> str:
    # You should calibrate these using your real measurements.
    if d_mm < 4.5:
        return "M4-ish"
    if d_mm < 5.5:
        return "M5-ish"
    if d_mm < 6.6:
        return "M6-ish"
    return "Unknown"


def measure_screw(
    bgr: np.ndarray,
    square_mm: float = 5.0,
    s_max: int = 85,
    v_min: int = 40,
    v_max: int = 245,
    debug: bool = False,
) -> tuple[ScrewMeasurement, np.ndarray]:
    px_per_mm = estimate_px_per_mm_from_squares(bgr, square_mm=square_mm)
    if px_per_mm is None:
        raise RuntimeError("Could not estimate px/mm from grid (try better lighting or more grid in view).")

    mask, squares, edges_clean = segment_screw_mask(bgr, s_max=s_max, v_min=v_min, v_max=v_max)
    
    mask_meas = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)


    c = best_screw_contour(mask_meas, min_area=1200.0, min_rot_ar=1.5)
    if c is None:
        cv2.imshow("mask_squares", squares)
        cv2.imshow("edges_clean", edges_clean)
        cv2.imshow("mask_screw", mask_meas)

        cv2.waitKey(0)
        raise RuntimeError("No screw-like contour detected (tune --s-max / --v-min / --v-max).")

    rect = cv2.minAreaRect(c)
    (rcx, rcy), (rw, rh), angle = rect

    # Rotate so long axis is horizontal
    if rw < rh:
        rot = angle
    else:
        rot = angle + 90.0

    rbgr, rmask = rotate_image_and_mask(bgr, mask_meas, rot)

    ys, xs = np.where(rmask > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    roi = rmask[y0:y1 + 1, x0:x1 + 1]
    roi = (roi > 0).astype(np.uint8) * 255

    prof, _, _ = width_profile(roi)
    if len(prof) < 40:
        raise RuntimeError("Screw too small in frame; move camera closer or increase resolution.")
    prof = smooth_1d(prof, k=11)

    nz = prof[prof > 0]
    if len(nz) < 20:
        raise RuntimeError("Profile extraction failed.")

    # Total length: use columns above a small threshold to ignore stray pixels
    shaft_guess = float(np.percentile(nz, 20))
    valid = prof > max(3.0, 0.25 * shaft_guess)
    idx = np.where(valid)[0]
    total_len_px = float(idx.max() - idx.min()) if len(idx) else float(len(prof) - 1)

    # Shaft diameter: take middle 50% region and lower percentile (reduces head influence)
    n = len(prof)
    mid = prof[int(0.25 * n): int(0.75 * n)]
    mid_nz = mid[mid > 0]
    shaft_px = float(np.percentile(mid_nz, 35))

    # Head diameter: high percentile overall
    head_px = float(np.percentile(nz, 95))

    # Determine head side and find under-head boundary by step in width
    left_mean = float(np.mean(prof[:max(8, n // 12)]))
    right_mean = float(np.mean(prof[-max(8, n // 12):]))
    head_on_left = left_mean > right_mean

    step_thresh = shaft_px * 1.35
    thick = np.where(prof > step_thresh)[0]

    under_head_px = None
    if len(thick) > 0:
        if head_on_left:
            cand = thick[thick < n // 2]
            if len(cand) > 0:
                boundary = int(np.max(cand))
                under_head_px = float((idx.max() if len(idx) else (n - 1)) - boundary)
        else:
            cand = thick[thick > n // 2]
            if len(cand) > 0:
                boundary = int(np.min(cand))
                under_head_px = float(boundary - (idx.min() if len(idx) else 0))

    total_length_mm = total_len_px / px_per_mm
    shaft_diam_mm = shaft_px / px_per_mm
    head_diam_mm = (head_px / px_per_mm) if head_px > (shaft_px * 1.20) else None
    under_head_mm = (under_head_px / px_per_mm) if under_head_px is not None else None

    # Confidence: based on elongation and head/shaft contrast
    rot_ar = max(rw, rh) / float(min(rw, rh) + 1e-6)
    contrast = (head_px - shaft_px) / max(shaft_px, 1.0)
    conf = float(np.clip(0.35 + 0.25 * min(rot_ar / 4.0, 1.0) + 0.40 * np.clip(contrast, 0.0, 1.0), 0.0, 1.0))

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

    if debug:
        # Show the actual segmentation mask that drives measurement
        cv2.imshow("mask_screw", mask_meas)
        cv2.imshow("mask_squares", white)


    meas = ScrewMeasurement(
        px_per_mm=px_per_mm,
        total_length_mm=total_length_mm,
        under_head_length_mm=under_head_mm,
        shaft_diam_mm=shaft_diam_mm,
        head_diam_mm=head_diam_mm,
        confidence=conf,
    )
    return meas, ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="ESP32 image URL, e.g. http://192.168.0.202/capture.jpg")
    ap.add_argument("--image", help="Path to local image")
    ap.add_argument("--square-mm", type=float, default=5.0, help="Grid square size in mm (default 5.0)")
    ap.add_argument("--s-max", type=int, default=85, help="Max saturation for metal mask")
    ap.add_argument("--v-min", type=int, default=40, help="Min value for metal mask")
    ap.add_argument("--v-max", type=int, default=245, help="Max value for metal mask")
    ap.add_argument("--save", help="Save annotated image path (optional)")
    ap.add_argument("--show", action="store_true", help="Show annotated image window")
    ap.add_argument("--debug", action="store_true", help="Show masks used for segmentation")
    args = ap.parse_args()

    if not args.url and not args.image:
        print("Provide --url or --image", file=sys.stderr)
        sys.exit(2)

    if args.url:
        bgr = cv2.cvtColor(fetch_image(args.url), cv2.COLOR_BGR2RGB)
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)  # normalize
    else:
        bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to read image file")

    meas, ann = measure_screw(
        bgr,
        square_mm=args.square_mm,
        s_max=args.s_max,
        v_min=args.v_min,
        v_max=args.v_max,
        debug=args.debug,
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

    if args.show:
        cv2.imshow("annotated", ann)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
