import argparse
import math
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np


# ISO 4762 / DIN 912 typical cap-head dimensions (mm)
# (Enough for M2..M10; extend if needed)
CAPHEAD_TABLE = {
    2.0:  {"d": 2.0,  "dk": 3.8,  "k": 2.0},
    2.5:  {"d": 2.5,  "dk": 4.5,  "k": 2.5},
    3.0:  {"d": 3.0,  "dk": 5.5,  "k": 3.0},
    4.0:  {"d": 4.0,  "dk": 7.0,  "k": 4.0},
    5.0:  {"d": 5.0,  "dk": 8.5,  "k": 5.0},
    6.0:  {"d": 6.0,  "dk": 10.0, "k": 6.0},
    8.0:  {"d": 8.0,  "dk": 13.0, "k": 8.0},
    10.0: {"d": 10.0, "dk": 16.0, "k": 10.0},
}


@dataclass
class Marker:
    corners: np.ndarray  # (4,2) float32
    center: Tuple[float, float]
    px_per_mm: float
    area: float


def imread_url(url: str, timeout: float = 5.0) -> Optional[np.ndarray]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def order_box_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def marker_side_px(corners: np.ndarray) -> float:
    p = order_box_points(corners)
    d01 = np.linalg.norm(p[0] - p[1])
    d12 = np.linalg.norm(p[1] - p[2])
    d23 = np.linalg.norm(p[2] - p[3])
    d30 = np.linalg.norm(p[3] - p[0])
    return float((d01 + d12 + d23 + d30) / 4.0)


def detect_markers(bgr: np.ndarray, marker_mm: float) -> Tuple[List[Marker], np.ndarray]:
    h, w = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    # Invert threshold so dark squares become white
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean noise
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    markers: List[Marker] = []

    if hierarchy is None:
        return [], th
    hierarchy = hierarchy[0]

    img_area = float(h * w)
    min_area = 0.002 * img_area
    max_area = 0.25 * img_area

    # Pass 1: prefer contours that have children (internal pattern holes)
    for require_child in (True, False):
        candidates: List[Marker] = []
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue

            has_child = (hierarchy[i][2] != -1)
            if require_child and not has_child:
                continue

            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), _ = rect
            rw, rh = float(rw), float(rh)
            if rw < 10 or rh < 10:
                continue
            aspect = max(rw, rh) / max(1e-6, min(rw, rh))
            if aspect > 1.25:
                continue

            box = cv2.boxPoints(rect).astype(np.float32)
            side = marker_side_px(box)
            extent = area / max(1.0, rw * rh)
            if extent < 0.55:
                continue

            pxmm = side / marker_mm
            candidates.append(Marker(corners=box, center=(cx, cy), px_per_mm=pxmm, area=area))

        candidates.sort(key=lambda m: m.area, reverse=True)
        markers = candidates[:2]
        if len(markers) >= 2 or (not require_child and len(markers) >= 1):
            break

    return markers, th


def build_marker_mask(shape: Tuple[int, int], markers: List[Marker]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for m in markers:
        poly = order_box_points(m.corners).astype(np.int32)
        cv2.fillConvexPoly(mask, poly, 255)
    # Expand to ensure we remove the whole marker + a little halo
    mask = cv2.dilate(mask, np.ones((11, 11), np.uint8), iterations=1)
    return mask


def fill_holes(bin_mask: np.ndarray) -> np.ndarray:
    h, w = bin_mask.shape
    flood = bin_mask.copy()
    tmp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, tmp, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(bin_mask, flood_inv)


def segment_screw(bgr: np.ndarray, marker_mask: np.ndarray,
                  min_len_mm: float, max_len_mm: float, pxmm_hint: float,
                  debug: bool = False) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Returns (screw_mask, info). screw_mask is uint8 (0/255).
    Uses multiple passes to reduce "sometimes works" behaviour.
    """
    h, w = bgr.shape[:2]
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Remove markers from consideration
    g_work = g.copy()
    g_work[marker_mask > 0] = 255

    img_area = float(h * w)
    best = None
    best_score = -1e18
    best_dbg = {}

    sigmas = [9, 13, 17]  # try a few background scales
    for sigma in sigmas:
        bg = cv2.GaussianBlur(g_work, (0, 0), sigmaX=sigma, sigmaY=sigma)
        diff = cv2.absdiff(g_work, bg)

        # Otsu threshold on diff
        _, bw = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean up
        k_close = max(3, int(round(pxmm_hint * 1.0)))
        k_open = max(3, int(round(pxmm_hint * 0.4)))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8), iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8), iterations=1)

        bw[marker_mask > 0] = 0

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 80 or area > 0.35 * img_area:
                continue

            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), _ = rect
            rw, rh = float(rw), float(rh)
            if rw < 5 or rh < 5:
                continue

            max_dim = max(rw, rh)
            min_dim = min(rw, rh)
            aspect = max_dim / max(1e-6, min_dim)

            # Rough length filter in mm (loose)
            len_mm_est = max_dim / max(1e-6, pxmm_hint)
            if len_mm_est < min_len_mm * 0.5 or len_mm_est > max_len_mm * 1.7:
                continue

            # Penalise border-touching blobs (paper edge / black mat)
            x, y, ww, hh = cv2.boundingRect(c)
            border = 0
            if x <= 1 or y <= 1 or (x + ww) >= (w - 2) or (y + hh) >= (h - 2):
                border = 1

            # Build mask for candidate
            cand = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cand, [c], -1, 255, thickness=-1)
            cand = fill_holes(cand)

            # Compute thickness stability (helps reject junk)
            cols = np.where(cand.sum(axis=0) > 0)[0]
            if len(cols) < 10:
                continue
            x0, x1 = int(cols[0]), int(cols[-1])
            L = max(1, x1 - x0 + 1)
            mid0 = x0 + int(0.25 * L)
            mid1 = x0 + int(0.75 * L)
            t = []
            for xx in range(mid0, mid1):
                ys = np.where(cand[:, xx] > 0)[0]
                if len(ys) > 0:
                    t.append(int(ys[-1] - ys[0] + 1))
            if len(t) < 10:
                continue
            t = np.array(t, dtype=np.float32)
            stability = float(np.std(t) / (np.mean(t) + 1e-6))

            # Scoring: prefer compact but stable elongated object
            score = (area ** 0.7) * (min(aspect, 8.0) ** 1.2) / (stability + 1e-3)
            if border:
                score *= 0.35

            if score > best_score:
                best_score = score
                best = cand
                best_dbg = {
                    "sigma": sigma,
                    "area": area,
                    "aspect": aspect,
                    "len_mm_est": len_mm_est,
                    "stability": stability,
                    "bbox": (x, y, ww, hh),
                }

    if best is None:
        return None, {"reason": "no suitable screw contour found"}

    return best, best_dbg


def pca_angle_from_mask(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return 0.0
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean = pts.mean(axis=0)
    pts0 = pts - mean
    cov = np.cov(pts0.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]
    ang = math.degrees(math.atan2(v[1], v[0]))
    return ang


def rotate_image_and_mask(bgr: np.ndarray, mask: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    bgr_r = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    mask_r = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return bgr_r, mask_r


def compute_thickness_profile(mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
    col_sum = (mask > 0).sum(axis=0).astype(np.int32)
    cols = np.where(col_sum > 0)[0]
    if len(cols) < 5:
        return np.array([], dtype=np.float32), 0, 0
    x0, x1 = int(cols[0]), int(cols[-1])

    t = np.zeros(mask.shape[1], dtype=np.float32)
    for x in range(x0, x1 + 1):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) > 0:
            t[x] = float(ys[-1] - ys[0] + 1)
    return t, x0, x1


def refine_endpoints(mask: np.ndarray, t: np.ndarray, x0: int, x1: int) -> Tuple[int, int]:
    # Ignore tiny “halo” at ends by requiring a minimum thickness
    mid = t[x0:x1 + 1]
    if len(mid) < 10:
        return x0, x1
    shaft_est = float(np.median(mid[mid > 0]))
    min_col = max(2.0, shaft_est * 0.25)

    # left
    xl = x0
    for x in range(x0, x1 + 1):
        if t[x] >= min_col:
            xl = x
            break
    # right
    xr = x1
    for x in range(x1, x0 - 1, -1):
        if t[x] >= min_col:
            xr = x
            break

    return int(xl), int(xr)


def choose_gauge_from_profile(pxmm: float, t: np.ndarray, x0: int, x1: int, gauges: List[float]) -> Tuple[float, Dict]:
    L = max(1, x1 - x0 + 1)
    # head side = side with larger thickness near the end
    edge_n = max(10, int(0.12 * L))
    left_peak = float(np.percentile(t[x0:x0 + edge_n], 95))
    right_peak = float(np.percentile(t[x1 - edge_n:x1 + 1], 95))
    head_on_left = left_peak >= right_peak

    if head_on_left:
        head_region = t[x0:x0 + edge_n]
    else:
        head_region = t[x1 - edge_n:x1 + 1]

    # Shaft region: central part, avoid ends
    mid0 = x0 + int(0.25 * L)
    mid1 = x0 + int(0.75 * L)
    shaft_region = t[mid0:mid1]
    shaft_px = float(np.median(shaft_region[shaft_region > 0])) if np.any(shaft_region > 0) else float(np.median(t[t > 0]))

    head_px = float(np.percentile(head_region[head_region > 0], 95)) if np.any(head_region > 0) else max(left_peak, right_peak)

    shaft_mm = shaft_px / max(1e-6, pxmm)
    head_mm = head_px / max(1e-6, pxmm)

    # Fit to table (prefer head diameter because it’s more separable at low resolution)
    best_g = gauges[0]
    best_e = 1e18
    for g in gauges:
        if g not in CAPHEAD_TABLE:
            continue
        exp_d = CAPHEAD_TABLE[g]["d"]
        exp_dk = CAPHEAD_TABLE[g]["dk"]
        e = (1.2 * (head_mm - exp_dk) ** 2) + (0.6 * (shaft_mm - exp_d) ** 2)
        if e < best_e:
            best_e = e
            best_g = g

    dbg = {
        "head_on_left": head_on_left,
        "shaft_px": shaft_px,
        "head_px": head_px,
        "shaft_mm": shaft_mm,
        "head_mm": head_mm,
        "fit_error": best_e,
    }
    return best_g, dbg


def blend_pxmm(markers: List[Marker], screw_center: Tuple[float, float]) -> float:
    if not markers:
        raise ValueError("No markers")
    if len(markers) == 1:
        return markers[0].px_per_mm
    sx, sy = screw_center
    ws = []
    vals = []
    for m in markers:
        mx, my = m.center
        d2 = (sx - mx) ** 2 + (sy - my) ** 2
        w = 1.0 / (d2 + 1.0)  # inverse distance^2 weighting
        ws.append(w)
        vals.append(m.px_per_mm)
    ws = np.array(ws, dtype=np.float32)
    vals = np.array(vals, dtype=np.float32)
    return float((ws * vals).sum() / (ws.sum() + 1e-6))


def snap(x: float, step: float) -> float:
    if step <= 0:
        return x
    return float(step * round(x / step))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Local image path")
    src.add_argument("--url", type=str, help="Image URL (e.g. ESP32 /capture.jpg)")

    ap.add_argument("--marker-mm", type=float, default=40.0)
    ap.add_argument("--min-length-mm", type=float, default=5.0)
    ap.add_argument("--max-length-mm", type=float, default=80.0)
    ap.add_argument("--length-step", type=float, default=5.0)
    ap.add_argument("--gauges", type=str, default="2,2.5,3,4,5,6")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save", type=str, default="")

    args = ap.parse_args()
    gauges = [float(x.strip()) for x in args.gauges.split(",") if x.strip()]

    if args.image:
        bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    else:
        bgr = imread_url(args.url)
    if bgr is None:
        raise SystemExit("Failed to load image")

    h, w = bgr.shape[:2]

    # 1) markers
    markers, marker_thresh = detect_markers(bgr, args.marker_mm)
    if len(markers) == 0:
        raise SystemExit("No markers detected")

    marker_mask = build_marker_mask((h, w), markers)
    pxmm_hint = float(np.median([m.px_per_mm for m in markers]))

    # 2) segment screw (multi-pass)
    screw_mask, seg_dbg = segment_screw(
        bgr=bgr,
        marker_mask=marker_mask,
        min_len_mm=args.min_length_mm,
        max_len_mm=args.max_length_mm,
        pxmm_hint=pxmm_hint,
        debug=args.debug,
    )
    if screw_mask is None:
        raise SystemExit(f"Measurement failed: {seg_dbg.get('reason','segmentation failed')}")

    # screw centroid for pxmm blending
    ys, xs = np.where(screw_mask > 0)
    cx = float(xs.mean()) if len(xs) else w / 2.0
    cy = float(ys.mean()) if len(ys) else h / 2.0
    pxmm = blend_pxmm(markers, (cx, cy))

    # 3) rotate to horizontal
    ang = pca_angle_from_mask(screw_mask)
    # rotate so major axis is ~horizontal
    bgr_r, mask_r = rotate_image_and_mask(bgr, screw_mask, -ang)

    # 4) profile & endpoints
    t, x0, x1 = compute_thickness_profile(mask_r)
    if t.size == 0:
        raise SystemExit("Measurement failed: empty thickness profile")

    xl, xr = refine_endpoints(mask_r, t, x0, x1)
    total_px = float(xr - xl + 1)

    # recompute profile now that endpoints are refined
    # (we just use the existing t; endpoints refined are enough)
    g_est, gauge_dbg = choose_gauge_from_profile(pxmm, t, xl, xr, gauges)

    if g_est not in CAPHEAD_TABLE:
        raise SystemExit(f"Gauge {g_est} missing from CAPHEAD_TABLE")

    head_h = float(CAPHEAD_TABLE[g_est]["k"])
    total_mm = total_px / max(1e-6, pxmm)
    under_mm = max(0.0, total_mm - head_h)

    under_snap = snap(under_mm, args.length_step)

    # 5) annotate
    ann = bgr.copy()
    # marker boxes
    for i, m in enumerate(markers):
        box = order_box_points(m.corners).astype(np.int32)
        cv2.polylines(ann, [box], True, (0, 255, 0), 2)
        cv2.putText(ann, f"marker{i+1}", (int(m.center[0]) - 20, int(m.center[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # screw contour
    contours, _ = cv2.findContours(screw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(ann, [c], -1, (255, 0, 0), 2)

    text = [
        f"Result: M{int(g_est) if g_est.is_integer() else g_est} x {int(under_snap)} (step={args.length_step}mm)",
        f"Gauge fit: head~{gauge_dbg['head_mm']:.2f}mm shaft~{gauge_dbg['shaft_mm']:.2f}mm  pxmm={pxmm:.3f}",
        f"Total length (end-to-end): {total_mm:.2f} mm",
        f"Underhead (total - head_h={head_h:.1f}): {under_mm:.2f} mm",
    ]
    y = 30
    for line in text:
        cv2.putText(ann, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 28

    if args.debug:
        print(f"M{g_est} x {under_snap}")
        print(f"pxmm={pxmm:.4f} total_mm={total_mm:.2f} under_mm={under_mm:.2f} head_h={head_h:.1f}")
        print(f"seg={seg_dbg}")
        print(f"gauge_dbg={gauge_dbg}")

    if args.save:
        cv2.imwrite(args.save, ann)
        print(f"Saved: {args.save}")

    if args.show:
        cv2.imshow("annotated", ann)
        cv2.imshow("marker_thresh", marker_thresh)
        cv2.imshow("screw_mask", screw_mask)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
