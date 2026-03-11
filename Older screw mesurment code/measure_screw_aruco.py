import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def save_dbg(debug_dir: Optional[str], name: str, img: np.ndarray) -> None:
    if not debug_dir:
        return
    ensure_dir(debug_dir)
    cv2.imwrite(os.path.join(debug_dir, name), img)

def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def quad_side_lengths(corners: np.ndarray) -> np.ndarray:
    c = corners.reshape(4, 2).astype(np.float32)
    d01 = np.linalg.norm(c[0] - c[1])
    d12 = np.linalg.norm(c[1] - c[2])
    d23 = np.linalg.norm(c[2] - c[3])
    d30 = np.linalg.norm(c[3] - c[0])
    return np.array([d01, d12, d23, d30], dtype=np.float32)

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class MarkerDetection:
    dict_name: str
    marker_id: int
    corners_px: np.ndarray  # (4,2)
    px_per_mm: float
    score: float


# -----------------------------
# ArUco detection (full-frame)
# -----------------------------

def _get_aruco_dict(dict_name: str):
    aruco = cv2.aruco
    if not hasattr(aruco, dict_name):
        raise ValueError(f"Unknown ArUco dict: {dict_name}")
    return aruco.getPredefinedDictionary(getattr(aruco, dict_name))

def _make_detector(dict_name: str):
    aruco = cv2.aruco
    dictionary = _get_aruco_dict(dict_name)

    params = aruco.DetectorParameters()

    # Robustness for small / low-contrast tags
    params.detectInvertedMarker = True

    # Wider adaptive threshold search
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 75
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant = 7

    # Accept smaller perimeters (small tags far away)
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0

    # More tolerant polygon approx / corner spacing
    params.polygonalApproxAccuracyRate = 0.06
    params.minCornerDistanceRate = 0.01
    params.minDistanceToBorder = 0  # IMPORTANT: allow tags near edges

    # Corner refinement
    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01

    # Newer OpenCV option (if available)
    if hasattr(params, "useAruco3Detection"):
        params.useAruco3Detection = True

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        return detector, dictionary, params
    else:
        return None, dictionary, params

def _detect_on_image(gray: np.ndarray, dict_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    aruco = cv2.aruco
    detector, dictionary, params = _make_detector(dict_name)
    if detector is not None:
        corners, ids, _rej = detector.detectMarkers(gray)
    else:
        corners, ids, _rej = aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None or len(ids) == 0:
        return None, None

    # corners is list of (1,4,2)
    ids = ids.reshape(-1).astype(int)
    corners_arr = np.array([c.reshape(4, 2) for c in corners], dtype=np.float32)
    return corners_arr, ids

def _preproc_variants(gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    variants: List[Tuple[str, np.ndarray]] = []

    variants.append(("gray", gray))

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    variants.append(("clahe", clahe.apply(gray)))

    # Equalize
    variants.append(("eq", cv2.equalizeHist(gray)))

    # Illumination normalization: subtract big blur
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    norm = cv2.normalize(cv2.subtract(gray, blur), None, 0, 255, cv2.NORM_MINMAX)
    variants.append(("norm", norm.astype(np.uint8)))

    # Slight blur to reduce noise (helps some tags)
    variants.append(("gblur", cv2.GaussianBlur(gray, (5, 5), 0)))

    # Add inverted variants too
    out: List[Tuple[str, np.ndarray]] = []
    for name, im in variants:
        out.append((name, im))
        out.append((name + "_inv", 255 - im))
    return out

def detect_marker_fullframe(
    bgr: np.ndarray,
    marker_mm: float,
    dicts: List[str],
    debug_dir: Optional[str] = None,
) -> Optional[MarkerDetection]:
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    save_dbg(debug_dir, "dbg_gray.png", gray0)

    H, W = gray0.shape[:2]
    variants = _preproc_variants(gray0)

    # A small scale pyramid helps when tags are tiny or blurred
    scales = [1.0, 1.5, 2.0, 2.5]

    best: Optional[MarkerDetection] = None

    for vname, gim in variants:
        for s in scales:
            if s == 1.0:
                g = gim
            else:
                g = cv2.resize(gim, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

            for dict_name in dicts:
                corners_arr, ids = _detect_on_image(g, dict_name)
                if ids is None:
                    continue

                for corners_px, mid in zip(corners_arr, ids):
                    # scale back corners to original pixel space
                    corners_orig = corners_px / float(s)

                    # Basic geometry checks
                    side = quad_side_lengths(order_quad(corners_orig))
                    side_mean = float(side.mean())
                    side_min = float(side.min())
                    side_max = float(side.max())

                    if side_mean < 25:   # too small -> likely noise
                        continue
                    if side_max / max(1e-6, side_min) > 1.35:
                        continue

                    # Score: prefer bigger + more square
                    squareness = 1.0 - clamp((side_max - side_min) / max(1e-6, side_mean), 0.0, 1.0)
                    size_score = clamp(side_mean / 200.0, 0.0, 1.0)  # saturates around 200px
                    score = 0.65 * squareness + 0.35 * size_score

                    px_per_mm = side_mean / float(marker_mm)

                    det = MarkerDetection(
                        dict_name=dict_name,
                        marker_id=int(mid),
                        corners_px=order_quad(corners_orig),
                        px_per_mm=float(px_per_mm),
                        score=float(score),
                    )

                    if (best is None) or (det.score > best.score):
                        best = det

    if best is not None:
        # Optional: refine corners with sub-pixel
        gray = gray0.copy()
        corners = best.corners_px.reshape(-1, 1, 2).astype(np.float32)
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
        best.corners_px = corners.reshape(4, 2)

        # Recompute px/mm with refined corners
        side = quad_side_lengths(best.corners_px)
        best.px_per_mm = float(side.mean() / float(marker_mm))

        # Debug overlay
        dbg = bgr.copy()
        cv2.polylines(dbg, [best.corners_px.astype(np.int32)], True, (0, 255, 0), 2)
        c = best.corners_px.mean(axis=0)
        cv2.putText(
            dbg,
            f"{best.dict_name} id={best.marker_id} px/mm={best.px_per_mm:.3f}",
            (int(c[0]) + 10, int(c[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        save_dbg(debug_dir, "dbg_marker_overlay.png", dbg)

    return best


# -----------------------------
# Screw segmentation + measurement
# -----------------------------

@dataclass
class ScrewMeasurement:
    length_mm: float
    diameter_mm: float
    center_px: Tuple[float, float]
    angle_deg: float
    bbox: Tuple[int, int, int, int]  # x,y,w,h


def segment_screw_mask(
    bgr: np.ndarray,
    marker_det: Optional[MarkerDetection],
    debug_dir: Optional[str] = None,
) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Illumination correction
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=21)
    norm = cv2.normalize(cv2.subtract(gray, blur), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold for dark objects (screw) on bright background
    norm_blur = cv2.GaussianBlur(norm, (5, 5), 0)
    _, bin_inv = cv2.threshold(norm_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    # Mask out marker region so it doesn't get picked as a "screw"
    if marker_det is not None:
        pad = 15
        x0 = int(max(0, marker_det.corners_px[:, 0].min() - pad))
        y0 = int(max(0, marker_det.corners_px[:, 1].min() - pad))
        x1 = int(min(bgr.shape[1] - 1, marker_det.corners_px[:, 0].max() + pad))
        y1 = int(min(bgr.shape[0] - 1, marker_det.corners_px[:, 1].max() + pad))
        bin_inv[y0:y1, x0:x1] = 0

    save_dbg(debug_dir, "dbg_norm.png", norm)
    save_dbg(debug_dir, "dbg_bin_inv.png", bin_inv)

    return bin_inv


def measure_screw_from_mask(
    mask: np.ndarray,
    px_per_mm: float,
    debug_dir: Optional[str] = None,
) -> Optional[ScrewMeasurement]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Pick the best elongated contour
    best = None
    best_score = -1.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (w, h), ang = rect
        long_side = max(w, h)
        short_side = min(w, h) + 1e-6
        aspect = long_side / short_side
        if aspect < 2.0:
            continue
        score = area * aspect
        if score > best_score:
            best_score = score
            best = (c, rect)

    if best is None:
        return None

    contour, rect = best
    (cx, cy), (w, h), ang = rect
    long_side = max(w, h)
    short_side = min(w, h)

    # PCA-based length/diameter (more stable than minAreaRect alone)
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean = pts.mean(axis=0)
    X = pts - mean
    cov = (X.T @ X) / max(1, len(X))
    eigvals, eigvecs = np.linalg.eigh(cov)
    u = eigvecs[:, np.argmax(eigvals)]  # major axis
    v = np.array([-u[1], u[0]], dtype=np.float32)  # perpendicular

    t = X @ u
    d = X @ v

    # Total tip-to-head length (pixel)
    tmin, tmax = float(t.min()), float(t.max())
    length_px = (tmax - tmin)

    # Estimate shank diameter using middle portion (avoid head)
    # Bin along t and find where width is stable/smallest
    nb = 40
    bins = np.linspace(tmin, tmax, nb + 1)
    widths = []
    centers = []
    for i in range(nb):
        lo, hi = bins[i], bins[i + 1]
        sel = (t >= lo) & (t < hi)
        if sel.sum() < 20:
            widths.append(np.nan)
            centers.append((lo + hi) * 0.5)
            continue
        dd = np.abs(d[sel])
        width = 2.0 * float(np.percentile(dd, 90))  # robust thickness
        widths.append(width)
        centers.append((lo + hi) * 0.5)

    widths = np.array(widths, dtype=np.float32)
    # take the lowest 30% widths as "shank"
    finite = np.isfinite(widths)
    if finite.sum() < 5:
        diameter_px = short_side
    else:
        ws = widths[finite]
        k = max(3, int(0.3 * len(ws)))
        shank_est = float(np.mean(np.sort(ws)[:k]))
        diameter_px = shank_est

    length_mm = float(length_px / px_per_mm)
    diameter_mm = float(diameter_px / px_per_mm)

    x, y, ww, hh = cv2.boundingRect(contour)
    angle_deg = float(np.degrees(np.arctan2(u[1], u[0])))

    # Debug plot mask with contour bbox
    if debug_dir:
        dbg = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(dbg, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(dbg, (x, y), (x + ww, y + hh), (255, 0, 0), 2)
        save_dbg(debug_dir, "dbg_screw_contour.png", dbg)

    return ScrewMeasurement(
        length_mm=length_mm,
        diameter_mm=diameter_mm,
        center_px=(float(cx), float(cy)),
        angle_deg=angle_deg,
        bbox=(int(x), int(y), int(ww), int(hh)),
    )


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--out", required=True, help="Output annotated image path")
    ap.add_argument("--debug-dir", default=None, help="Write debug images here")
    ap.add_argument("--marker-mm", type=float, required=True, help="ArUco black-square side length in mm (NOT the white margin)")
    ap.add_argument("--dict", dest="dicts", action="append", default=["DICT_4X4_50"],
                    help="ArUco dictionary name (repeatable). Default: DICT_4X4_50")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"Could not read image: {args.image}")
        raise SystemExit(2)

    ensure_dir(args.debug_dir)

    marker = detect_marker_fullframe(bgr, args.marker_mm, args.dicts, args.debug_dir)

    result: Dict = {
        "found_marker": marker is not None,
        "dict": marker.dict_name if marker else None,
        "marker_id": marker.marker_id if marker else None,
        "px_per_mm": marker.px_per_mm if marker else None,
        "found_screw": False,
        "screw": None,
        "annotated_image": os.path.abspath(args.out),
        "debug_dir": os.path.abspath(args.debug_dir) if args.debug_dir else None,
    }

    ann = bgr.copy()

    if marker is not None:
        cv2.polylines(ann, [marker.corners_px.astype(np.int32)], True, (0, 255, 0), 2)
        c = marker.corners_px.mean(axis=0)
        cv2.putText(
            ann,
            f"{marker.dict_name} id={marker.marker_id} px/mm={marker.px_per_mm:.3f}",
            (int(c[0]) + 10, int(c[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        mask = segment_screw_mask(ann, marker, args.debug_dir)
        meas = measure_screw_from_mask(mask, marker.px_per_mm, args.debug_dir)

        if meas is not None:
            result["found_screw"] = True
            result["screw"] = {
                "length_mm": round(meas.length_mm, 2),
                "diameter_mm": round(meas.diameter_mm, 2),
                "center_px": [round(meas.center_px[0], 1), round(meas.center_px[1], 1)],
                "angle_deg": round(meas.angle_deg, 1),
                "bbox": list(meas.bbox),
            }

            # annotate
            x, y, w, h = meas.bbox
            cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx, cy = int(meas.center_px[0]), int(meas.center_px[1])
            cv2.circle(ann, (cx, cy), 4, (0, 255, 0), -1)

            cv2.putText(
                ann,
                f"L={meas.length_mm:.1f}mm D={meas.diameter_mm:.1f}mm",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    else:
        # Still save debug hint image
        save_dbg(args.debug_dir, "dbg_no_marker.png", ann)

    cv2.imwrite(args.out, ann)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
