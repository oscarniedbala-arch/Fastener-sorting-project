#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Utilities
# ----------------------------

@dataclass
class Detection:
    dict_name: str
    ids: np.ndarray            # shape (N,1)
    corners: List[np.ndarray]  # list of (1,4,2) float arrays in FULL IMAGE coordinates
    score: float               # heuristic confidence score


def _aruco_available() -> bool:
    return hasattr(cv2, "aruco")


def _make_params():
    # OpenCV API differs slightly between versions
    aruco = cv2.aruco
    try:
        params = aruco.DetectorParameters()
    except Exception:
        params = aruco.DetectorParameters_create()

    # Helpful tweaks for real-world prints / backlit paper
    if hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = True

    # Loosen a little, but not too much
    if hasattr(params, "cornerRefinementMethod"):
        params.cornerRefinementMethod = getattr(aruco, "CORNER_REFINE_SUBPIX", 1)
    if hasattr(params, "minDistanceToBorder"):
        params.minDistanceToBorder = 1
    if hasattr(params, "minCornerDistanceRate"):
        params.minCornerDistanceRate = 0.01
    if hasattr(params, "polygonalApproxAccuracyRate"):
        params.polygonalApproxAccuracyRate = 0.05
    if hasattr(params, "minMarkerPerimeterRate"):
        params.minMarkerPerimeterRate = 0.02
    if hasattr(params, "maxMarkerPerimeterRate"):
        params.maxMarkerPerimeterRate = 4.0

    # Adaptive threshold window sweep (moderate)
    if hasattr(params, "adaptiveThreshWinSizeMin"):
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 4

    return params


def _detect_once(gray: np.ndarray, dict_name: str) -> Tuple[List[np.ndarray], Optional[np.ndarray], List[np.ndarray]]:
    aruco = cv2.aruco
    if not hasattr(aruco, dict_name):
        return [], None, []

    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    params = _make_params()

    # Newer OpenCV has ArucoDetector; older uses detectMarkers directly
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=params)

    return corners, ids, rejected


def _score_detection(corners: List[np.ndarray], ids: np.ndarray, img_shape: Tuple[int, int]) -> float:
    # Heuristic: prefer more markers + larger perimeter + nearer center
    h, w = img_shape
    n = 0 if ids is None else len(ids)
    if n == 0:
        return 0.0

    perims = []
    centers = []
    for c in corners:
        pts = c.reshape(4, 2)
        per = float(cv2.arcLength(pts.astype(np.float32), True))
        perims.append(per)
        centers.append(pts.mean(axis=0))

    mean_per = float(np.mean(perims)) if perims else 0.0
    center = np.mean(np.array(centers), axis=0) if centers else np.array([w / 2, h / 2], dtype=np.float32)
    dist = float(np.linalg.norm(center - np.array([w / 2, h / 2], dtype=np.float32)))

    # Higher is better
    return (n * 1000.0) + (mean_per * 2.0) - (dist * 0.5)


def _preprocess_variants(gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    variants = [("gray", gray)]

    # CLAHE helps when paper is backlit / washed out
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g2 = clahe.apply(gray)
    variants.append(("clahe", g2))

    # Slight blur can stabilize thresholding / edge noise
    variants.append(("blur", cv2.GaussianBlur(gray, (5, 5), 0)))
    variants.append(("clahe_blur", cv2.GaussianBlur(g2, (5, 5), 0)))

    # Inverted variants for backlit situations
    out = []
    for name, im in variants:
        out.append((name, im))
        out.append((name + "_inv", 255 - im))
    return out


def _dict_candidates() -> List[str]:
    # Include ArUco original + common families + AprilTags (often mistaken/used interchangeably)
    return [
        "DICT_ARUCO_ORIGINAL",
        "DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_7X7_50",
        "DICT_4X4_100", "DICT_5X5_100", "DICT_6X6_100", "DICT_7X7_100",
        "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
    ]


def _find_square_rois(gray: np.ndarray, max_rois: int = 6) -> List[Tuple[int, int, int, int]]:
    """
    Finds square-ish dark regions (likely marker border) and returns ROIs as (x0,y0,x1,y1).
    Designed for backlit paper and vignetting.
    """
    h, w = gray.shape

    # Threshold for dark objects (marker border). Use percentile to adapt across lighting.
    t = int(np.percentile(gray, 25))
    t = max(30, min(140, t))
    mask = (gray < t).astype(np.uint8) * 255

    # Clean up noise
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    cx0, cy0 = w / 2.0, h / 2.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 600:  # ignore tiny blobs
            continue

        x, y, ww, hh = cv2.boundingRect(c)

        # Skip stuff touching borders (vignetting)
        if x <= 2 or y <= 2 or (x + ww) >= (w - 2) or (y + hh) >= (h - 2):
            continue

        ar = ww / float(hh + 1e-6)
        if not (0.75 <= ar <= 1.33):
            continue
        if ww < 40 or hh < 40:
            continue

        center = np.array([x + ww / 2.0, y + hh / 2.0], dtype=np.float32)
        dist = float(np.linalg.norm(center - np.array([cx0, cy0], dtype=np.float32)))

        # Score: larger + closer to center
        score = area - (dist * 40.0)
        cands.append((score, x, y, ww, hh))

    cands.sort(reverse=True, key=lambda z: z[0])
    rois = []

    for _, x, y, ww, hh in cands[:max_rois]:
        # Expand ROI generously to include quiet zone around marker
        margin = int(round(max(ww, hh) * 0.70))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(w, x + ww + margin)
        y1 = min(h, y + hh + margin)
        rois.append((x0, y0, x1, y1))

    return rois


def detect_best(gray: np.ndarray, dicts: List[str], debug_dir: Optional[str] = None) -> Optional[Detection]:
    h, w = gray.shape

    # 1) Try full image with preprocess variants
    best: Optional[Detection] = None
    for vname, im in _preprocess_variants(gray):
        for dn in dicts:
            corners, ids, _rej = _detect_once(im, dn)
            if ids is None or len(ids) == 0:
                continue
            sc = _score_detection(corners, ids, (h, w))
            det = Detection(dn, ids, corners, sc)
            if best is None or det.score > best.score:
                best = det

    if best is not None:
        return best

    # 2) If nothing found, find candidate ROIs and retry (this is the big reliability gain)
    rois = _find_square_rois(gray)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "debug_gray.png"), gray)
        # Visualize ROIs
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x0, y0, x1, y1) in rois:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "debug_rois.png"), vis)

    for i, (x0, y0, x1, y1) in enumerate(rois):
        roi = gray[y0:y1, x0:x1]
        # Upscale to help detector when marker is small in the overall frame
        scale = 3.0
        roi_up = cv2.resize(roi, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        for vname, im in _preprocess_variants(roi_up):
            for dn in dicts:
                corners, ids, _rej = _detect_once(im, dn)
                if ids is None or len(ids) == 0:
                    continue

                # Convert ROI-up corners back to full-image coordinates
                fixed_corners = []
                for c in corners:
                    pts = c.reshape(4, 2) / scale
                    pts[:, 0] += x0
                    pts[:, 1] += y0
                    fixed_corners.append(pts.reshape(1, 4, 2))

                sc = _score_detection(fixed_corners, ids, (h, w))
                det = Detection(dn, ids, fixed_corners, sc)
                if best is None or det.score > best.score:
                    best = det

                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, f"debug_roi_{i}_{vname}.png"), im)

    return best


def marker_px_per_mm(corners_1: np.ndarray, marker_size_mm: float) -> float:
    """
    corners_1: shape (1,4,2) in full image coords.
    """
    pts = corners_1.reshape(4, 2).astype(np.float32)
    # side lengths
    d01 = np.linalg.norm(pts[1] - pts[0])
    d12 = np.linalg.norm(pts[2] - pts[1])
    d23 = np.linalg.norm(pts[3] - pts[2])
    d30 = np.linalg.norm(pts[0] - pts[3])
    side_px = float(np.mean([d01, d12, d23, d30]))
    return side_px / float(marker_size_mm)


def draw_annotated(bgr: np.ndarray, det: Detection, marker_size_mm: Optional[float] = None) -> np.ndarray:
    out = bgr.copy()
    for idx, (c, mid) in enumerate(zip(det.corners, det.ids.flatten().tolist()), start=1):
        pts = c.reshape(4, 2).astype(int)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        center = tuple(np.round(pts.mean(axis=0)).astype(int))
        label = f"{idx}: {det.dict_name} id={mid}"
        if marker_size_mm:
            ppm = marker_px_per_mm(c, marker_size_mm)
            label += f"  px/mm={ppm:.3f}"
        cv2.putText(out, label, (center[0] - 140, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(out, center, 4, (0, 255, 0), -1)
    return out


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image (jpg/png)")
    ap.add_argument("--out", default="aruco_annotated.png", help="Output annotated image path")
    ap.add_argument("--debug-dir", default="", help="Optional debug directory to write intermediate images")
    ap.add_argument("--dict", default="", help="Force a single dictionary name (e.g., DICT_ARUCO_ORIGINAL)")
    ap.add_argument("--marker-mm", type=float, default=0.0, help="Known marker side length in mm (for px/mm)")
    ap.add_argument("--show", action="store_true", help="Show window")
    args = ap.parse_args()

    if not _aruco_available():
        raise SystemExit(
            "cv2.aruco is not available. Install opencv-contrib-python in your venv:\n"
            "  python -m pip uninstall -y opencv-python\n"
            "  python -m pip install opencv-contrib-python\n"
        )

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    dicts = [args.dict] if args.dict else _dict_candidates()

    debug_dir = args.debug_dir.strip() or None
    det = detect_best(gray, dicts, debug_dir=debug_dir)

    result = {
        "image": os.path.abspath(args.image),
        "found": bool(det),
        "annotated_image": os.path.abspath(args.out),
    }

    if det is None:
        cv2.imwrite(args.out, bgr)
        result.update({"count": 0, "dictionaries_tried": dicts})
        print(json.dumps(result, indent=2))
        return

    annotated = draw_annotated(bgr, det, marker_size_mm=(args.marker_mm if args.marker_mm > 0 else None))
    cv2.imwrite(args.out, annotated)

    result.update({
        "count": int(len(det.ids)),
        "dictionary": det.dict_name,
        "ids": det.ids.flatten().tolist(),
        "corners_px": [
            [[float(x), float(y)] for (x, y) in c.reshape(4, 2)]
            for c in det.corners
        ],
    })

    if args.marker_mm > 0 and len(det.corners) > 0:
        ppm = marker_px_per_mm(det.corners[0], args.marker_mm)
        result["px_per_mm"] = float(ppm)

    print(json.dumps(result, indent=2))

    if args.show:
        cv2.imshow("aruco", annotated)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
