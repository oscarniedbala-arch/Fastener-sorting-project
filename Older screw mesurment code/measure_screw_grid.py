import argparse
import math
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
    """
    Detect the white squares and estimate their median width/height in pixels.
    Returns px/mm. If detection fails, returns None.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # White-ish squares: low saturation + high value
    mask = cv2.inRange(hsv, (0, 0, 160), (180, 90, 255))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    widths = []
    heights = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 80 or area > 30000:
            continue
        ar = w / float(h + 1e-6)
        if ar < 0.70 or ar > 1.45:
            continue
        widths.append(w)
        heights.append(h)

    if not widths:
        return None

    square_px = float(np.median([(w + h) / 2.0 for w, h in zip(widths, heights)]))
    return square_px / square_mm


def largest_elongated_contour(bgr: np.ndarray, min_area: float = 500.0, min_rot_ar: float = 1.4) -> np.ndarray | None:
    """
    Edge-based segmentation; choose best elongated contour (rotation-invariant).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 30, 120)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = bgr.shape[:2]
    frame_area = float(h * w)

    best = None
    best_score = -1.0

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue

        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), _ = rect
        if rw < 5 or rh < 5:
            continue
        rot_ar = max(rw, rh) / float(min(rw, rh) + 1e-6)
        if rot_ar < min_rot_ar:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        if (bw * bh) > 0.45 * frame_area:
            continue

        score = area * min(rot_ar, 8.0)
        if score > best_score:
            best_score = score
            best = c

    return best


def rotate_image_and_mask(bgr: np.ndarray, mask: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate around center; expand canvas to avoid cropping.
    """
    h, w = bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # compute new bounds
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    rbgr = cv2.warpAffine(bgr, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rmask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rbgr, rmask


def width_profile_along_x(bin_mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    For each x in bbox, count non-zero pixels in that column.
    """
    ys, xs = np.where(bin_mask > 0)
    if len(xs) == 0:
        return np.array([]), 0, 0
    x0, x1 = int(xs.min()), int(xs.max())
    prof = np.zeros(x1 - x0 + 1, dtype=np.float32)
    for i, x in enumerate(range(x0, x1 + 1)):
        prof[i] = float(np.count_nonzero(bin_mask[:, x]))
    return prof, x0, x1


def smooth_1d(a: np.ndarray, k: int = 9) -> np.ndarray:
    if len(a) < k:
        return a
    k = max(3, k | 1)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(a, kernel, mode="same")


def measure_screw(bgr: np.ndarray, square_mm: float = 5.0) -> tuple[ScrewMeasurement, np.ndarray]:
    """
    Returns measurement + annotated image.
    """
    px_per_mm = estimate_px_per_mm_from_squares(bgr, square_mm=square_mm)
    if px_per_mm is None:
        raise RuntimeError("Could not estimate px/mm from grid. Improve lighting/contrast or keep grid centered.")

    c = largest_elongated_contour(bgr, min_area=600.0, min_rot_ar=1.35)
    if c is None:
        raise RuntimeError("No screw-like contour detected.")

    # Build filled mask from contour
    mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)

    rect = cv2.minAreaRect(c)
    (rcx, rcy), (rw, rh), angle = rect

    # OpenCV minAreaRect angle conventions: rotate so major axis is horizontal
    if rw < rh:
        rot = angle
    else:
        rot = angle + 90.0

    rbgr, rmask = rotate_image_and_mask(bgr, mask, rot)

    # Extract bbox of rotated mask
    ys, xs = np.where(rmask > 0)
    if len(xs) == 0:
        raise RuntimeError("Rotated mask empty.")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    roi_mask = rmask[y0:y1 + 1, x0:x1 + 1]
    roi_mask = (roi_mask > 0).astype(np.uint8) * 255

    prof, px0, px1 = width_profile_along_x(roi_mask)
    if len(prof) < 20:
        raise RuntimeError("Profile too short; screw likely too small/far away.")

    prof = smooth_1d(prof, k=11)
    nonzero = prof[prof > 0]

    # Shaft width: use a low percentile of the width profile (robust to head)
    shaft_px = float(np.percentile(nonzero, 25))
    head_px = float(np.percentile(nonzero, 95))

    # Determine which end has the head: compare average width near ends
    n = len(prof)
    left_mean = float(np.mean(prof[:max(5, n // 12)]))
    right_mean = float(np.mean(prof[-max(5, n // 12):]))
    head_on_left = left_mean > right_mean

    # Under-head boundary detection: where width rises above threshold
    thresh = shaft_px * 1.35
    idxs = np.where(prof > thresh)[0]

    under_head_px = None
    if len(idxs) > 0:
        if head_on_left:
            # boundary is the last "thick" index near left side -> use max idx within first half
            candidates = idxs[idxs < n // 2]
            if len(candidates) > 0:
                boundary = int(np.max(candidates))
                under_head_px = float((n - 1) - boundary)
        else:
            candidates = idxs[idxs > n // 2]
            if len(candidates) > 0:
                boundary = int(np.min(candidates))
                under_head_px = float(boundary)

    total_length_px = float(n - 1)

    total_length_mm = total_length_px / px_per_mm
    shaft_diam_mm = shaft_px / px_per_mm
    head_diam_mm = head_px / px_per_mm if head_px > (shaft_px * 1.15) else None
    under_head_mm = (under_head_px / px_per_mm) if under_head_px is not None else None

    # Simple confidence: how elongated and how strong the head/shaft contrast is
    contrast = min(1.0, max(0.0, (head_px - shaft_px) / max(shaft_px, 1.0)))
    conf = 0.5 + 0.5 * contrast
    conf = float(np.clip(conf, 0.0, 1.0))

    # Annotate original image (draw contour + rect)
    ann = bgr.copy()
    cv2.drawContours(ann, [c], -1, (0, 0, 255), 2)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(ann, [box], 0, (255, 0, 0), 2)

    txt1 = f"px/mm={px_per_mm:.2f}"
    txt2 = f"L_total={total_length_mm:.1f}mm  D_shaft~{shaft_diam_mm:.2f}mm"
    txt3 = f"L_under_head={under_head_mm:.1f}mm" if under_head_mm else "L_under_head=(n/a)"
    cv2.putText(ann, txt1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(ann, txt1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(ann, txt2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(ann, txt2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(ann, txt3, (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(ann, txt3, (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

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
    ap.add_argument("--save", help="Save annotated image path (optional)")
    ap.add_argument("--show", action="store_true", help="Show annotated image window")
    args = ap.parse_args()

    if not args.url and not args.image:
        print("Provide --url or --image", file=sys.stderr)
        sys.exit(2)

    if args.url:
        bgr = fetch_image(args.url)
    else:
        bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to read image file")

    meas, ann = measure_screw(bgr, square_mm=args.square_mm)

    print(f"px_per_mm:           {meas.px_per_mm:.3f}")
    print(f"total_length_mm:     {meas.total_length_mm:.2f}")
    if meas.under_head_length_mm is not None:
        print(f"under_head_length_mm:{meas.under_head_length_mm:.2f}")
    else:
        print("under_head_length_mm:(n/a)")
    print(f"shaft_diam_mm:       {meas.shaft_diam_mm:.3f}")
    if meas.head_diam_mm is not None:
        print(f"head_diam_mm:        {meas.head_diam_mm:.3f}")
    else:
        print("head_diam_mm:        (n/a)")
    print(f"confidence:          {meas.confidence:.2f}")

    if args.save:
        cv2.imwrite(args.save, ann)
        print(f"Saved annotated: {args.save}")

    if args.show:
        cv2.imshow("annotated", ann)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
