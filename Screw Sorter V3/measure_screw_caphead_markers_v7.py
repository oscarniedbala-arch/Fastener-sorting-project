import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def parse_gauges(s: str) -> List[float]:
    out = []
    for part in s.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return sorted(out)


def nearest_metric(value: float, gauges: List[float]) -> Tuple[float, float]:
    """Return (nearest_gauge, signed_error_mm)."""
    if not gauges:
        return value, 0.0
    best = min(gauges, key=lambda g: abs(value - g))
    return best, (value - best)


def snap(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(x / step) * step


def mm_to_px(mm: float, px_per_mm: float) -> int:
    return int(round(mm * px_per_mm))


def safe_int(x: float) -> int:
    return int(max(0, round(x)))


@dataclass
class Marker:
    box_pts: np.ndarray  # 4x2 float
    cx: float
    cy: float
    side_px: float


def order_box_points(pts: np.ndarray) -> np.ndarray:
    """Return box points in consistent order: tl, tr, br, bl."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def contour_min_area_rect(cnt) -> Tuple[np.ndarray, float, float]:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = order_box_points(box)
    (w, h) = rect[1]
    return box, float(w), float(h)


# -----------------------------
# Marker detection
# -----------------------------
def find_two_markers(img_bgr: np.ndarray, debug: bool = False) -> Tuple[Marker, Marker, np.ndarray]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Robust threshold for dark squares on light background
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 5
    )
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500:  # reject tiny noise
            continue
        box, bw, bh = contour_min_area_rect(c)
        side = max(bw, bh)
        short = min(bw, bh)
        if short <= 0:
            continue
        aspect = side / short

        # must be roughly square
        if aspect > 1.25:
            continue

        # markers are typically in lower half; keep some flexibility
        cx = float(box[:, 0].mean())
        cy = float(box[:, 1].mean())
        candidates.append((area, side, cx, cy, box))

    if len(candidates) < 2:
        raise RuntimeError("Could not find 2 marker candidates")

    # Prefer larger, squarer, lower-ish objects
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    top = candidates[:10]

    # Choose best pair separated in X
    best_pair = None
    best_score = -1e9
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a1, s1, x1, y1, b1 = top[i]
            a2, s2, x2, y2, b2 = top[j]
            dx = abs(x1 - x2)
            if dx < w * 0.25:
                continue
            # score: big area + separation - mismatch
            score = (a1 + a2) + 0.5 * dx - 500.0 * abs(s1 - s2)
            if score > best_score:
                best_score = score
                best_pair = (top[i], top[j])

    if best_pair is None:
        # fallback to top 2
        best_pair = (top[0], top[1])

    (a1, s1, x1, y1, b1), (a2, s2, x2, y2, b2) = best_pair
    m1 = Marker(box_pts=b1, cx=x1, cy=y1, side_px=s1)
    m2 = Marker(box_pts=b2, cx=x2, cy=y2, side_px=s2)

    # enforce left/right
    left, right = (m1, m2) if m1.cx < m2.cx else (m2, m1)
    return left, right, thr


# -----------------------------
# Screw segmentation in constrained ROI
# -----------------------------
def build_roi_mask(img_shape, left: Marker, right: Marker, px_per_mm: float,
                   roi_margin_mm: float, y_margin_mm: float) -> np.ndarray:
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # marker bounds
    lxs = left.box_pts[:, 0]
    rxs = right.box_pts[:, 0]
    lys = left.box_pts[:, 1]
    rys = right.box_pts[:, 1]

    l_right = float(np.max(lxs))
    r_left = float(np.min(rxs))
    m_bottom = float(max(np.max(lys), np.max(rys)))

    mx = mm_to_px(roi_margin_mm, px_per_mm)
    my = mm_to_px(y_margin_mm, px_per_mm)

    x0 = safe_int(l_right + mx)
    x1 = safe_int(r_left - mx)
    y0 = 0
    y1 = safe_int(m_bottom - my)  # keep above marker bottoms (you confirmed screws are above)

    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w, x1))
    y1 = max(0, min(h, y1))

    if x1 <= x0 or y1 <= y0:
        # fallback: keep between marker centers and allow more y
        x0 = safe_int(min(left.cx, right.cx))
        x1 = safe_int(max(left.cx, right.cx))
        y1 = safe_int(m_bottom)

    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    # subtract marker quads so they can never be selected
    cv2.fillConvexPoly(mask, left.box_pts.astype(np.int32), 0)
    cv2.fillConvexPoly(mask, right.box_pts.astype(np.int32), 0)
    return mask


def segment_screw_edges(img_bgr: np.ndarray, roi_mask: np.ndarray,
                        px_per_mm: float, edge_dilate_mm: float, close_mm: float) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # local contrast helps metal edges
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    # apply ROI
    g_roi = cv2.bitwise_and(g, g, mask=roi_mask)

    # auto canny thresholds
    med = float(np.median(g_roi[roi_mask > 0])) if np.any(roi_mask > 0) else float(np.median(g_roi))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))

    edges = cv2.Canny(g_roi, lo, hi)

    # thicken edges a bit
    dpx = max(1, mm_to_px(edge_dilate_mm, px_per_mm))
    edges = cv2.dilate(edges, np.ones((dpx, dpx), np.uint8), iterations=1)

    # close gaps
    cpx = max(3, mm_to_px(close_mm, px_per_mm))
    if cpx % 2 == 0:
        cpx += 1
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((cpx, cpx), np.uint8), iterations=1)

    # fill contours to get solid blobs
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(edges)
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        cv2.drawContours(filled, [c], -1, 255, thickness=-1)

    # keep only within ROI mask
    filled = cv2.bitwise_and(filled, filled, mask=roi_mask)
    return filled


def pick_best_screw_component(mask: np.ndarray, left: Marker, right: Marker,
                             marker_mm: float, px_per_mm: float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pick the best screw component using strong geometric + positional constraints."""
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        raise RuntimeError("No components found in screw mask")

    marker_area_px = (marker_mm * px_per_mm) ** 2  # approx marker side^2 in px^2

    best_i = None
    best_score = -1e18

    x_min_allowed = min(left.cx, right.cx)
    x_max_allowed = max(left.cx, right.cx)
    y_max_allowed = max(left.cy, right.cy)  # screw must be above marker centers (strong bias)

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]

        # hard-ish position constraints
        if not (x_min_allowed <= cx <= x_max_allowed):
            continue
        if cy > y_max_allowed:
            continue

        if area < 150:
            continue

        # reject marker-sized blobs
        if area > 0.45 * marker_area_px:
            continue

        # elongation check using bbox
        aspect = (max(w, h) / max(1, min(w, h)))
        if aspect < 1.8:
            continue

        # score favors elongation and area (but not huge)
        score = (aspect * 5000.0) + area - 2.0 * (w * h)

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        raise RuntimeError("No suitable screw component found (constraints too strict or segmentation failed)")

    x, y, w, h, area = stats[best_i]
    comp = (labels == best_i).astype(np.uint8) * 255
    return comp, (int(x), int(y), int(w), int(h))


# -----------------------------
# Measurement
# -----------------------------
def pca_axis(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    pts0 = pts - mean
    cov = np.cov(pts0.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    return mean, axis


def rotate_mask(mask: np.ndarray, center: Tuple[float, float], angle_deg: float) -> np.ndarray:
    h, w = mask.shape[:2]
    M = cv2.getRotationMatrix2D((center[0], center[1]), angle_deg, 1.0)
    rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rot


def measure_caphead(mask: np.ndarray, px_per_mm: float) -> dict:
    mean, axis = pca_axis(mask)

    # angle to rotate so axis becomes horizontal
    angle = math.degrees(math.atan2(axis[1], axis[0]))
    rot = rotate_mask(mask, (mean[0], mean[1]), -angle)

    ys, xs = np.where(rot > 0)
    x0, x1 = xs.min(), xs.max()
    total_len_px = float(x1 - x0 + 1)

    # thickness per column
    thickness = []
    for x in range(x0, x1 + 1):
        col = rot[:, x]
        idx = np.where(col > 0)[0]
        if idx.size == 0:
            continue
        thickness.append(float(idx.max() - idx.min() + 1))

    if len(thickness) < 10:
        raise RuntimeError("Mask too small / too few columns to measure")

    thickness = np.array(thickness, dtype=np.float32)

    # Estimate shaft thickness using a low percentile (robust to head being thicker)
    shaft_px = float(np.percentile(thickness, 25.0))
    head_px = float(np.percentile(thickness, 85.0))

    total_mm = total_len_px / px_per_mm
    shaft_mm = shaft_px / px_per_mm
    head_mm = head_px / px_per_mm

    return {
        "angle_deg": angle,
        "total_len_px": total_len_px,
        "shaft_px": shaft_px,
        "head_px": head_px,
        "total_mm": total_mm,
        "shaft_mm": shaft_mm,
        "head_mm": head_mm,
        "rot_mask": rot,
        "bbox_rot_x0x1": (int(x0), int(x1)),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="Input image path")
    g.add_argument("--url", type=str, help="Image URL")

    ap.add_argument("--marker-mm", type=float, default=40.0)
    ap.add_argument("--gauges", type=str, default="2,2.5,3,4,5,6")
    ap.add_argument("--length-step", type=float, default=5.0)

    # ROI / segmentation controls
    ap.add_argument("--roi-margin-mm", type=float, default=3.0, help="Horizontal margin from markers in mm")
    ap.add_argument("--y-margin-mm", type=float, default=2.0, help="How far above marker bottoms ROI ends (mm)")
    ap.add_argument("--edge-dilate-mm", type=float, default=1.2)
    ap.add_argument("--close-mm", type=float, default=1.8)

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save", type=str, default="", help="Save annotated image path")

    args = ap.parse_args()
    gauges = parse_gauges(args.gauges)

    if args.url:
        cap = cv2.VideoCapture(args.url)
        ok, img = cap.read()
        cap.release()
        if not ok or img is None:
            raise RuntimeError(f"Failed to read URL: {args.url}")
    else:
        img = cv2.imread(args.image)
        if img is None:
            raise RuntimeError(f"Failed to read image: {args.image}")

    left, right, marker_thr = find_two_markers(img, debug=args.debug)
    px_per_mm = float((left.side_px + right.side_px) * 0.5 / args.marker_mm)

    roi_mask = build_roi_mask(img.shape, left, right, px_per_mm, args.roi_margin_mm, args.y_margin_mm)
    screw_mask = segment_screw_edges(img, roi_mask, px_per_mm, args.edge_dilate_mm, args.close_mm)
    screw_comp, bbox = pick_best_screw_component(screw_mask, left, right, args.marker_mm, px_per_mm)

    meas = measure_caphead(screw_comp, px_per_mm)
    gauge_mm_est = meas["shaft_mm"]
    gauge_fit, gauge_err = nearest_metric(gauge_mm_est, gauges)

    # Caphead assumption: head height k ~= D
    total_mm = meas["total_mm"]
    under_mm = total_mm - gauge_fit

    under_snap = snap(under_mm, args.length_step)

    # Annotation
    ann = img.copy()
    cv2.polylines(ann, [left.box_pts.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(ann, [right.box_pts.astype(np.int32)], True, (0, 255, 0), 2)

    x, y, w, h = bbox
    cv2.rectangle(ann, (x, y), (x + w, y + h), (255, 0, 0), 2)

    txt = [
        f"Result: M{gauge_fit:g} x {under_snap:g} (step={args.length_step:g}mm)",
        f"px/mm={px_per_mm:.3f}  total={total_mm:.2f}mm  under={under_mm:.2f}mm",
        f"shaft_est={meas['shaft_mm']:.2f}mm (err {gauge_err:+.2f}) head_px_p85={meas['head_mm']:.2f}mm",
    ]
    y0 = 30
    for i, t in enumerate(txt):
        cv2.putText(ann, t, (10, y0 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    print(f"M{gauge_fit:g} x {under_snap:g}")
    if args.debug:
        dbg = {
            "px_per_mm": px_per_mm,
            "bbox": bbox,
            "meas": {k: meas[k] for k in ["total_mm", "shaft_mm", "head_mm", "total_len_px", "shaft_px", "head_px"]},
        }
        print("debug:", dbg)

    if args.save:
        cv2.imwrite(args.save, ann)
        print("Saved:", args.save)

    if args.show:
        cv2.imshow("marker_thresh", marker_thr)
        cv2.imshow("roi_mask", roi_mask)
        cv2.imshow("screw_mask", screw_mask)
        cv2.imshow("screw_comp", screw_comp)
        cv2.imshow("annotated", ann)
        cv2.waitKey(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Measurement failed:", e)
        sys.exit(2)
