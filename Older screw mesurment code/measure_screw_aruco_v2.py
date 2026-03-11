import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np


# ----------------------------
# Helpers / debug
# ----------------------------
def ensure_dir(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    os.makedirs(p, exist_ok=True)
    return p


def imwrite(debug_dir: Optional[str], name: str, img: np.ndarray) -> None:
    if not debug_dir:
        return
    path = os.path.join(debug_dir, name)
    cv2.imwrite(path, img)


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def clahe(gray: np.ndarray) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return c.apply(gray)


def sharpen(gray: np.ndarray) -> np.ndarray:
    # mild unsharp mask
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    out = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return out


# ----------------------------
# ArUco dictionary brute-force
# ----------------------------
def list_predefined_aruco_dicts() -> List[str]:
    # OpenCV exposes many DICT_* constants; we brute-force those that are dictionaries.
    names = []
    for k in dir(cv2.aruco):
        if k.startswith("DICT_"):
            names.append(k)
    # Keep a stable ordering
    names.sort()
    return names


@dataclass
class MarkerDetection:
    dict_name: str
    marker_id: int
    corners: np.ndarray  # shape (4,2) float32
    score: float         # heuristic score (higher better)


def detect_marker_best(
    img_bgr: np.ndarray,
    dict_candidates: List[str],
    debug_dir: Optional[str] = None,
) -> Optional[MarkerDetection]:
    aruco = cv2.aruco
    gray0 = to_gray(img_bgr)

    # Preprocessing variants (in practice these cover most real-world failures)
    variants: List[Tuple[str, np.ndarray]] = [
        ("gray", gray0),
        ("clahe", clahe(gray0)),
        ("sharp", sharpen(gray0)),
        ("clahe_sharp", sharpen(clahe(gray0))),
    ]

    # Detector parameters tuned for low-quality / low-contrast feeds
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 7
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 75
    params.adaptiveThreshWinSizeStep = 6
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.03
    params.minCornerDistanceRate = 0.02
    params.minDistanceToBorder = 2
    params.minOtsuStdDev = 3.0
    params.perspectiveRemoveIgnoredMarginPerCell = 0.1
    params.maxErroneousBitsInBorderRate = 0.6
    params.errorCorrectionRate = 0.8
    params.detectInvertedMarker = True  # critical when lighting inverts contrast

    best: Optional[MarkerDetection] = None

    for vname, gray in variants:
        imwrite(debug_dir, f"dbg_{vname}.png", gray)

        # Try a light denoise to reduce JPEG blocking without killing edges
        gray_dn = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        for dict_name in dict_candidates:
            if not hasattr(aruco, dict_name):
                continue
            dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
            detector = aruco.ArucoDetector(dictionary, params)

            corners_list, ids, rejected = detector.detectMarkers(gray_dn)

            if ids is None or len(ids) == 0:
                continue

            # Evaluate candidates: prefer larger markers and “squarer” geometry
            for corners, mid in zip(corners_list, ids.flatten()):
                c = corners.reshape(4, 2).astype(np.float32)

                # side lengths
                s0 = np.linalg.norm(c[1] - c[0])
                s1 = np.linalg.norm(c[2] - c[1])
                s2 = np.linalg.norm(c[3] - c[2])
                s3 = np.linalg.norm(c[0] - c[3])
                perim = s0 + s1 + s2 + s3
                s_mean = perim / 4.0
                squareness = 1.0 - (np.std([s0, s1, s2, s3]) / (s_mean + 1e-6))
                squareness = float(np.clip(squareness, 0.0, 1.0))

                # area proxy (bigger is better)
                area = float(abs(cv2.contourArea(c.reshape(-1, 1, 2))))

                # score heuristic
                score = (area ** 0.5) * (0.3 + 0.7 * squareness)

                cand = MarkerDetection(dict_name=dict_name, marker_id=int(mid), corners=c, score=score)
                if best is None or cand.score > best.score:
                    best = cand

    return best


# ----------------------------
# Screw segmentation + measurement in marker plane (homography)
# ----------------------------
def order_corners(c: np.ndarray) -> np.ndarray:
    # Ensure consistent order: tl, tr, br, bl
    # c shape (4,2)
    s = c.sum(axis=1)
    d = np.diff(c, axis=1).reshape(-1)
    tl = c[np.argmin(s)]
    br = c[np.argmax(s)]
    tr = c[np.argmin(d)]
    bl = c[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def measure_screw_with_homography(
    img_bgr: np.ndarray,
    marker_corners_px: np.ndarray,
    marker_mm: float,
    debug_dir: Optional[str] = None,
) -> Optional[Dict]:
    gray = to_gray(img_bgr)

    # Homography: pixel -> mm (marker coordinate frame)
    src = order_corners(marker_corners_px)
    dst = np.array([[0, 0], [marker_mm, 0], [marker_mm, marker_mm], [0, marker_mm]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    # Build a mask to exclude marker region from screw segmentation
    marker_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(marker_mask, src.astype(np.int32), 255)

    # Segment screw: adaptive threshold + morphology; exclude marker
    g = clahe(gray)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    # In many scenes screws are darker than background; try both and pick the better contour later
    th1 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 51, 5)
    th2 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 51, 5)

    # Remove marker area from both
    th1[marker_mask > 0] = 0
    th2[marker_mask > 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel, iterations=1)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel, iterations=2)

    th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel, iterations=1)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel, iterations=2)

    imwrite(debug_dir, "dbg_bin_inv.png", th1)
    imwrite(debug_dir, "dbg_bin.png", th2)

    def best_contour(bin_img: np.ndarray) -> Optional[np.ndarray]:
        cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        # choose largest plausible elongated object
        bestc = None
        bestscore = -1.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 300:  # reject tiny noise
                continue
            x, y, w, h = cv2.boundingRect(c)
            ar = max(w, h) / (min(w, h) + 1e-6)
            # prefer elongated (screw) but allow moderate
            score = float(area) * float(min(ar, 10.0))
            if score > bestscore:
                bestscore = score
                bestc = c
        return bestc

    c1 = best_contour(th1)
    c2 = best_contour(th2)

    if c1 is None and c2 is None:
        imwrite(debug_dir, "dbg_no_screw.png", img_bgr)
        return None

    # pick the contour that yields larger mm-length after homography
    def contour_len_mm(cnt: np.ndarray) -> float:
        pts = cnt.reshape(-1, 2).astype(np.float32)
        pts_mm = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        # PCA in mm space
        mean = pts_mm.mean(axis=0)
        X = pts_mm - mean
        cov = np.cov(X.T)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, np.argmax(vals)]
        t = X @ axis
        return float(np.percentile(t, 99) - np.percentile(t, 1))

    candidates = [(c1, contour_len_mm(c1)) if c1 is not None else (None, -1),
                  (c2, contour_len_mm(c2)) if c2 is not None else (None, -1)]
    cnt, _ = max(candidates, key=lambda x: x[1])
    if cnt is None:
        return None

    # Transform contour to mm-space and measure via PCA (robust against perspective)
    pts = cnt.reshape(-1, 2).astype(np.float32)
    pts_mm = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)

    mean = pts_mm.mean(axis=0)
    X = pts_mm - mean
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, np.argmax(vals)]
    minor = vecs[:, np.argmin(vals)]

    t_major = X @ major
    t_minor = X @ minor

    length_mm = float(np.percentile(t_major, 99) - np.percentile(t_major, 1))
    diam_mm = float(np.percentile(t_minor, 95) - np.percentile(t_minor, 5))

    # endpoints in mm (for overlay)
    t0 = float(np.percentile(t_major, 1))
    t1 = float(np.percentile(t_major, 99))
    p0_mm = mean + major * t0
    p1_mm = mean + major * t1

    # Convert endpoints back to pixels for drawing: use inverse homography
    Hinv = np.linalg.inv(H)
    p0_px = cv2.perspectiveTransform(np.array([[p0_mm]], dtype=np.float32), Hinv).reshape(2)
    p1_px = cv2.perspectiveTransform(np.array([[p1_mm]], dtype=np.float32), Hinv).reshape(2)

    return {
        "length_mm": length_mm,
        "diam_mm": diam_mm,
        "p0_px": [float(p0_px[0]), float(p0_px[1])],
        "p1_px": [float(p1_px[0]), float(p1_px[1])],
    }


def classify_metric(d_mm: float) -> str:
    # Basic mapping for common metric screws (tweak thresholds to your collection)
    if d_mm < 2.4:
        return "M2"
    if d_mm < 3.4:
        return "M3"
    if d_mm < 4.4:
        return "M4"
    if d_mm < 5.4:
        return "M5"
    if d_mm < 6.5:
        return "M6"
    return "unknown"


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image (jpg/png)")
    ap.add_argument("--out", default="measured.png", help="Output annotated image path")
    ap.add_argument("--debug-dir", default=None, help="Write debug images here")
    ap.add_argument("--marker-mm", type=float, required=True, help="Printed marker side length in mm (BLACK square size)")
    ap.add_argument("--dict", action="append", default=[], help="Optional dict(s). If omitted: auto-scan all DICT_*.")
    args = ap.parse_args()

    debug_dir = ensure_dir(args.debug_dir)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read image: {args.image}")
        raise SystemExit(2)

    dicts = args.dict if args.dict else list_predefined_aruco_dicts()

    best = detect_marker_best(img, dicts, debug_dir=debug_dir)

    out = {
        "found_marker": best is not None,
        "dict": None,
        "marker_id": None,
        "marker_mm": args.marker_mm,
        "found_screw": False,
        "screw": None,
        "annotated_image": os.path.abspath(args.out),
        "debug_dir": os.path.abspath(debug_dir) if debug_dir else None,
    }

    vis = img.copy()

    if best is None:
        imwrite(debug_dir, "dbg_no_marker.png", vis)
        cv2.imwrite(args.out, vis)
        print(json.dumps(out, indent=2))
        return

    out["dict"] = best.dict_name
    out["marker_id"] = best.marker_id

    c = best.corners.astype(np.int32)
    cv2.polylines(vis, [c.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    cx, cy = c.mean(axis=0).astype(int)
    cv2.putText(vis, f"{best.dict_name} id={best.marker_id}",
                (max(10, cx - 180), max(30, cy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    screw = measure_screw_with_homography(vis, best.corners, args.marker_mm, debug_dir=debug_dir)
    if screw is not None:
        out["found_screw"] = True
        out["screw"] = screw
        L = screw["length_mm"]
        D = screw["diam_mm"]
        cls = classify_metric(D)
        p0 = tuple(map(int, screw["p0_px"]))
        p1 = tuple(map(int, screw["p1_px"]))
        cv2.line(vis, p0, p1, (0, 255, 0), 2)
        cv2.putText(vis, f"{cls}  L={L:.1f}mm  D={D:.1f}mm",
                    (max(10, cx - 180), min(vis.shape[0] - 10, cy + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        imwrite(debug_dir, "dbg_screw_contour_overlay.png", vis)

    cv2.imwrite(args.out, vis)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
