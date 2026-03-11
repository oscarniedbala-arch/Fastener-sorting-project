import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def imwrite(path: str, img: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order corners as [tl, tr, br, bl]. pts shape (4,2)."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = (pts[:, 0] - pts[:, 1])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def mean_side_length_px(corners: np.ndarray) -> float:
    corners = corners.reshape(4, 2).astype(np.float32)
    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    return float(np.mean(sides))

def squareness(corners: np.ndarray) -> float:
    corners = corners.reshape(4, 2).astype(np.float32)
    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    mn, mx = float(min(sides)), float(max(sides))
    if mx <= 1e-6:
        return 0.0
    return mn / mx

def touches_border(rect: Tuple[int, int, int, int], w: int, h: int, margin: int = 2) -> bool:
    x, y, rw, rh = rect
    return (x <= margin) or (y <= margin) or (x + rw >= w - margin) or (y + rh >= h - margin)

def clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

def sharpen(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    # Divide by a heavy blur to flatten illumination gradients
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    bg = np.clip(bg, 1, 255).astype(np.uint8)
    norm = cv2.divide(gray, bg, scale=255)
    return norm

def pca_major_axis(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    points_xy: (N,2) float32
    Returns (center, unit_major_axis)
    """
    pts = points_xy.astype(np.float32)
    mean = np.mean(pts, axis=0)
    X = pts - mean
    cov = (X.T @ X) / max(len(pts), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    major = major / (np.linalg.norm(major) + 1e-9)
    return mean, major

def nearest_metric_thread(d_mm: float) -> str:
    # Coarse mapping; you can tighten tolerances once your pipeline is stable.
    sizes = [
        ("M2", 2.0),
        ("M2.5", 2.5),
        ("M3", 3.0),
        ("M4", 4.0),
        ("M5", 5.0),
        ("M6", 6.0),
        ("M8", 8.0),
        ("M10", 10.0),
    ]
    best = min(sizes, key=lambda s: abs(d_mm - s[1]))
    # If it's wildly off, return unknown
    if abs(d_mm - best[1]) > 1.2:
        return "unknown"
    return best[0]


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class MarkerCandidate:
    dict_name: str
    marker_id: int
    corners_px: np.ndarray  # (4,2) ordered tl,tr,br,bl
    side_px: float
    square: float
    area_px2: float
    score: float


# -----------------------------
# Marker detection
# -----------------------------

def list_all_aruco_dicts() -> List[str]:
    
    preferred = [
        "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
        "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
        "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
        "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
        "DICT_ARUCO_ORIGINAL",
        # AprilTags 
        "DICT_APRILTAG_16h5",
        "DICT_APRILTAG_25h9",
        "DICT_APRILTAG_36h10",
        "DICT_APRILTAG_36h11",
    ]
    avail = []
    aruco = cv2.aruco
    for name in preferred:
        if hasattr(aruco, name):
            avail.append(name)
    return avail

def get_predefined_dictionary(dict_name: str):
    aruco = cv2.aruco
    return aruco.getPredefinedDictionary(getattr(aruco, dict_name))

def detect_best_marker(
    gray: np.ndarray,
    dict_names: List[str],
    min_marker_side_px: int,
    min_perimeter_rate: float,
    expected_ids: Optional[List[int]],
    debug_dir: Optional[str] = None
) -> Optional[MarkerCandidate]:

    aruco = cv2.aruco

    params = aruco.DetectorParameters()
    params.detectInvertedMarker = True  
    params.minMarkerPerimeterRate = float(min_perimeter_rate)
    params.maxMarkerPerimeterRate = 4.0
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 10
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.05

    preprocesses = [
        ("gray", gray),
        ("clahe", clahe(gray)),
        ("sharp", sharpen(gray)),
        ("clahe_sharp", sharpen(clahe(gray))),
        ("norm", normalize_illumination(gray)),
        ("norm_clahe", clahe(normalize_illumination(gray))),
    ]

    all_candidates: List[MarkerCandidate] = []

    for pname, g in preprocesses:
        # Mild denoise
        gdn = cv2.fastNlMeansDenoising(g, None, 10, 7, 21)

        for dn in dict_names:
            dictionary = get_predefined_dictionary(dn)
            detector = aruco.ArucoDetector(dictionary, params)
            corners_list, ids, _ = detector.detectMarkers(gdn)

            if ids is None or len(ids) == 0:
                continue

            for corners, mid in zip(corners_list, ids.flatten()):
                mid = int(mid)
                if expected_ids is not None and mid not in expected_ids:
                    continue

                c = order_corners(corners.reshape(4, 2))
                side = mean_side_length_px(c)
                sq = squareness(c)
                area = float(cv2.contourArea(c.astype(np.float32)))

                # HARD REJECTION of tiny false positives
                if side < float(min_marker_side_px):
                    continue
                if sq < 0.70:
                    continue
                if area < 1000.0:
                    continue

                score = area * (sq ** 2)

                all_candidates.append(
                    MarkerCandidate(
                        dict_name=dn,
                        marker_id=mid,
                        corners_px=c,
                        side_px=side,
                        square=sq,
                        area_px2=area,
                        score=score,
                    )
                )

        if all_candidates:
            break

    if debug_dir:
        ensure_dir(debug_dir)
        dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cand in sorted(all_candidates, key=lambda x: x.score, reverse=True)[:10]:
            pts = cand.corners_px.astype(int).reshape(-1, 1, 2)
            cv2.polylines(dbg, [pts], True, (0, 255, 0), 2)
            txt = f"{cand.dict_name} id={cand.marker_id} side={cand.side_px:.0f}px sq={cand.square:.2f}"
            x, y = int(cand.corners_px[0, 0]), int(cand.corners_px[0, 1])
            cv2.putText(dbg, txt, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        imwrite(os.path.join(debug_dir, "dbg_marker_candidates.png"), dbg)

    if not all_candidates:
        return None

    best = max(all_candidates, key=lambda c: c.score)
    return best


# -----------------------------
# Metric warp (pixel -> mm plane)
# -----------------------------

def build_metric_warp(
    img_bgr: np.ndarray,
    marker: MarkerCandidate,
    marker_mm: float,
    warp_px_per_mm: float,
    debug_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
      warped_bgr, M (pixel->warped_px homography), px_per_mm_global_in_warp
    """
    h, w = img_bgr.shape[:2]

    src = marker.corners_px.astype(np.float32)
    dst_mm = np.array([[0, 0], [marker_mm, 0], [marker_mm, marker_mm], [0, marker_mm]], dtype=np.float32)

    H_px_to_mm = cv2.getPerspectiveTransform(src, dst_mm)

    img_corners = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float32)
    corners_mm = cv2.perspectiveTransform(img_corners, H_px_to_mm).reshape(-1, 2)

    minx, miny = np.min(corners_mm, axis=0)
    maxx, maxy = np.max(corners_mm, axis=0)

    margin_mm = 5.0
    minx -= margin_mm
    miny -= margin_mm
    maxx += margin_mm
    maxy += margin_mm

    out_w = int(math.ceil((maxx - minx) * warp_px_per_mm))
    out_h = int(math.ceil((maxy - miny) * warp_px_per_mm))

    out_w = int(np.clip(out_w, 200, 3000))
    out_h = int(np.clip(out_h, 200, 3000))

    A = np.array([
        [warp_px_per_mm, 0, -minx * warp_px_per_mm],
        [0, warp_px_per_mm, -miny * warp_px_per_mm],
        [0, 0, 1],
    ], dtype=np.float32)

    M = A @ H_px_to_mm

    warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR)

    if debug_dir:
        dbg = warped.copy()
        src_pts = marker.corners_px.reshape(-1, 1, 2).astype(np.float32)
        warped_pts = cv2.perspectiveTransform(src_pts, M).reshape(-1, 2).astype(int)
        cv2.polylines(dbg, [warped_pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        imwrite(os.path.join(debug_dir, "dbg_warped.png"), dbg)

    # In warped image, px_per_mm is exactly warp_px_per_mm by construction
    return warped, M, float(warp_px_per_mm)


# -----------------------------
# Screw detection + measurement in warped metric image
# -----------------------------

def find_best_screw_contour(
    warped_bgr: np.ndarray,
    marker_warp_poly: np.ndarray,
    px_per_mm: float,
    screw_len_range_mm: Tuple[float, float],
    screw_d_range_mm: Tuple[float, float],
    debug_dir: Optional[str] = None
) -> Optional[dict]:
    """
    Returns dict with screw contour and measurements, or None.
    """
    g0 = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    g0 = normalize_illumination(g0)
    g0 = cv2.GaussianBlur(g0, (5, 5), 0)

    h, w = g0.shape[:2]

    # Mask out the marker region
    marker_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(marker_mask, marker_warp_poly.astype(int), 255)
    marker_mask = cv2.dilate(marker_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)), iterations=1)

    # Build a working mask area
    valid_mask = np.ones((h, w), dtype=np.uint8) * 255
    valid_mask[marker_mask > 0] = 0

    candidates = []

    def evaluate_binary(bin_img: np.ndarray, tag: str):
        bin2 = cv2.bitwise_and(bin_img, bin_img, mask=valid_mask)

        bin2 = cv2.morphologyEx(bin2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        bin2 = cv2.morphologyEx(bin2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)

        if debug_dir:
            imwrite(os.path.join(debug_dir, f"dbg_bin_{tag}.png"), bin2)

        cnts, _ = cv2.findContours(bin2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 300:  # too tiny
                continue

            x, y, rw, rh = cv2.boundingRect(c)
            if touches_border((x, y, rw, rh), w, h, margin=3):
                continue

            rect = cv2.minAreaRect(c)
            (cx, cy), (rw2, rh2), ang = rect
            length_px = max(rw2, rh2)
            width_px = min(rw2, rh2)

            if width_px <= 1.0:
                continue

            length_mm = length_px / px_per_mm
            width_mm = width_px / px_per_mm

            if not (screw_len_range_mm[0] <= length_mm <= screw_len_range_mm[1]):
                continue
            if not (screw_d_range_mm[0] <= width_mm <= screw_d_range_mm[1]):
                continue

            aspect = length_px / width_px
            # Score: prefer long, thin, large area
            score = float(area) * (aspect ** 1.2)

            candidates.append((score, c, tag, rect))

    _, t_otsu = cv2.threshold(g0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_otsu_inv = cv2.bitwise_not(t_otsu)
    evaluate_binary(t_otsu, "otsu")
    evaluate_binary(t_otsu_inv, "otsu_inv")

    # Adaptive (sometimes better on ESP32 gradients)
    t_adp = cv2.adaptiveThreshold(g0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 2)
    t_adp_inv = cv2.bitwise_not(t_adp)
    evaluate_binary(t_adp, "adp")
    evaluate_binary(t_adp_inv, "adp_inv")

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_contour, best_tag, best_rect = candidates[0]

    # Refine measurements using PCA endpoints
    pts = best_contour.reshape(-1, 2).astype(np.float32)
    center, axis = pca_major_axis(pts)

    proj = (pts - center) @ axis
    pmin, pmax = float(np.min(proj)), float(np.max(proj))
    length_px = pmax - pmin
    length_mm = length_px / px_per_mm

    # Diameter: use orthogonal spread, robustly, ignoring the head by taking mid 60% of the axis
    orth = np.array([-axis[1], axis[0]], dtype=np.float32)
    t = (proj - pmin) / max((pmax - pmin), 1e-6)
    mid_pts = pts[(t > 0.2) & (t < 0.8)]
    if len(mid_pts) < 50:
        mid_pts = pts

    oproj = (mid_pts - center) @ orth
    lo, hi = np.percentile(oproj, [10, 90])
    diam_px = float(hi - lo)
    diam_mm = diam_px / px_per_mm

    p1 = center + axis * pmin
    p2 = center + axis * pmax

    return {
        "contour": best_contour,
        "method": best_tag,
        "length_mm": float(length_mm),
        "diam_mm": float(diam_mm),
        "axis_p1_px": [float(p1[0]), float(p1[1])],
        "axis_p2_px": [float(p2[0]), float(p2[1])],
        "center_px": [float(center[0]), float(center[1])],
    }


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Measure screw length/diameter using an ArUco marker scale reference.")
    ap.add_argument("--image", required=True, help="Path to input image (jpg/png).")
    ap.add_argument("--out", default="measured.png", help="Annotated output image path.")
    ap.add_argument("--debug-dir", default=None, help="If set, writes debug images here.")
    ap.add_argument("--marker-mm", type=float, required=True, help="Marker BLACK square side length in mm (not including quiet zone).")
    ap.add_argument("--warp-px-per-mm", type=float, default=10.0, help="Warp resolution in px/mm (10 is a good start).")
    ap.add_argument("--min-marker-side-px", type=int, default=80,
                    help="Reject marker detections smaller than this (prevents false positives).")
    ap.add_argument("--min-perimeter-rate", type=float, default=0.08,
                    help="OpenCV detector minMarkerPerimeterRate. Raise to reduce tiny false positives.")
    ap.add_argument("--dict", action="append", default=None,
                    help="Dictionary name(s), e.g. --dict DICT_4X4_50. Repeatable. If omitted, uses a safe preferred set.")
    ap.add_argument("--scan-all", action="store_true",
                    help="Scan ALL OpenCV DICT_* dictionaries (more tolerant, but can increase false positives).")
    ap.add_argument("--expected-id", action="append", type=int, default=None,
                    help="Optional: restrict to expected marker ID(s). Repeatable.")
    ap.add_argument("--screw-len-min", type=float, default=5.0)
    ap.add_argument("--screw-len-max", type=float, default=120.0)
    ap.add_argument("--screw-d-min", type=float, default=1.0)
    ap.add_argument("--screw-d-max", type=float, default=12.0)
    args = ap.parse_args()

    ensure_dir(args.debug_dir)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read image: {args.image}")
        raise SystemExit(2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if not hasattr(cv2, "aruco"):
        raise SystemExit("cv2.aruco not available. Install opencv-contrib-python.")

    if args.scan_all:
        dict_names = [k for k in dir(cv2.aruco) if k.startswith("DICT_")]
    else:
        dict_names = args.dict if args.dict else list_all_aruco_dicts()

    marker = detect_best_marker(
        gray=gray,
        dict_names=dict_names,
        min_marker_side_px=args.min_marker_side_px,
        min_perimeter_rate=args.min_perimeter_rate,
        expected_ids=args.expected_id,
        debug_dir=args.debug_dir,
    )

    result = {
        "found_marker": False,
        "dict": None,
        "marker_id": None,
        "px_per_mm": None,
        "found_screw": False,
        "screw": None,
        "annotated_image": os.path.abspath(args.out),
        "debug_dir": os.path.abspath(args.debug_dir) if args.debug_dir else None,
    }

    annotated = img.copy()

    if marker is None:
        imwrite(args.out, annotated)
        print(json.dumps(result, indent=2))
        return

    # Marker sanity: compute px/mm from marker pixel side / marker_mm
    px_per_mm_from_marker = marker.side_px / float(args.marker_mm)
    result["found_marker"] = True
    result["dict"] = marker.dict_name
    result["marker_id"] = marker.marker_id
    result["px_per_mm"] = float(px_per_mm_from_marker)

    # Draw marker on original
    cv2.polylines(annotated, [marker.corners_px.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    cv2.putText(
        annotated,
        f"{marker.dict_name} id={marker.marker_id} px/mm={px_per_mm_from_marker:.3f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Build warp
    warped, M, px_per_mm = build_metric_warp(
        img_bgr=img,
        marker=marker,
        marker_mm=float(args.marker_mm),
        warp_px_per_mm=float(args.warp_px_per_mm),
        debug_dir=args.debug_dir,
    )

    marker_pts = marker.corners_px.reshape(-1, 1, 2).astype(np.float32)
    marker_warp = cv2.perspectiveTransform(marker_pts, M).reshape(-1, 2)

    screw = find_best_screw_contour(
        warped_bgr=warped,
        marker_warp_poly=marker_warp,
        px_per_mm=float(px_per_mm),
        screw_len_range_mm=(args.screw_len_min, args.screw_len_max),
        screw_d_range_mm=(args.screw_d_min, args.screw_d_max),
        debug_dir=args.debug_dir,
    )

    if screw is None:
        if args.debug_dir:
            dbg = warped.copy()
            cv2.polylines(dbg, [marker_warp.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            imwrite(os.path.join(args.debug_dir, "dbg_warped_no_screw.png"), dbg)

        imwrite(args.out, annotated)
        print(json.dumps(result, indent=2))
        return

    # Classification
    thread = nearest_metric_thread(screw["diam_mm"])
    result["found_screw"] = True
    result["screw"] = {
        "thread": thread,
        "length_mm": round(screw["length_mm"], 2),
        "diam_mm": round(screw["diam_mm"], 2),
        "method": screw["method"],
    }

    # Draw screw measurement in warped debug
    if args.debug_dir:
        dbg = warped.copy()
        cnt = screw["contour"]
        cv2.drawContours(dbg, [cnt], -1, (0, 255, 0), 2)

        p1 = tuple(int(v) for v in screw["axis_p1_px"])
        p2 = tuple(int(v) for v in screw["axis_p2_px"])
        cv2.line(dbg, p1, p2, (0, 255, 0), 2)

        txt = f"{thread} L={screw['length_mm']:.1f}mm D={screw['diam_mm']:.1f}mm ({screw['method']})"
        cv2.putText(dbg, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        imwrite(os.path.join(args.debug_dir, "dbg_warped_screw.png"), dbg)

    # simple overlay
    cv2.putText(
        annotated,
        f"{thread} L={screw['length_mm']:.1f}mm D={screw['diam_mm']:.1f}mm",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    imwrite(args.out, annotated)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
