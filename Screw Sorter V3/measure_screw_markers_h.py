import argparse
import math
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np


# ----------------------------
# IO
# ----------------------------

def imread_url(url: str) -> np.ndarray:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = np.frombuffer(resp.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image from URL")
    return img


# ----------------------------
# Marker detection (square outer border)
# ----------------------------

@dataclass
class Marker:
    corners: np.ndarray  # (4,2) float32: tl,tr,br,bl
    area: float
    center: Tuple[float, float]
    px_per_mm: float


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_square_markers(bgr: np.ndarray, marker_mm: float,
                          min_area_px: int = 2500,
                          close_kernel: int = 9,
                          max_rect_ratio: float = 1.30) -> Tuple[List[Marker], np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers: List[Marker] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < float(min_area_px):
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        corners = order_points(approx.reshape(-1, 2))
        sides = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
        if min(sides) < 5:
            continue
        rect_ratio = max(sides) / max(min(sides), 1e-6)
        if rect_ratio > max_rect_ratio:
            continue

        side_px = float(np.mean(sides))
        px_per_mm = side_px / float(marker_mm)
        cx, cy = float(corners[:, 0].mean()), float(corners[:, 1].mean())
        markers.append(Marker(corners=corners, area=area, center=(cx, cy), px_per_mm=px_per_mm))

    markers.sort(key=lambda m: m.area, reverse=True)
    return markers, th


def homography_from_marker(marker: Marker, marker_mm: float) -> np.ndarray:
    src = marker.corners.astype(np.float32)
    dst = np.array([[0, 0], [marker_mm, 0], [marker_mm, marker_mm], [0, marker_mm]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


# ----------------------------
# Preprocess + candidate selection (rotation-invariant)
# ----------------------------

def preprocess_blob(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    v = float(np.median(g))
    lower = int(max(0, 0.50 * v))
    upper = int(min(255, 1.50 * v))
    edges = cv2.Canny(g, lower, upper)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blob = cv2.dilate(edges, k, iterations=2)
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, k, iterations=2)
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, k, iterations=1)
    return edges, blob


def rotated_rect_features(contour):
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), ang = rect
    w = max(float(w), 1e-6)
    h = max(float(h), 1e-6)
    long_side = max(w, h)
    short_side = min(w, h)
    aspect = long_side / short_side
    return rect, aspect, long_side, short_side


def pick_best_screw_contour(bgr: np.ndarray, blob: np.ndarray,
                            markers: List[Marker],
                            min_length_mm: float,
                            max_length_mm: float,
                            marker_mm: float) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float]]]:
    H, W = blob.shape
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Use marker px/mm (median) only to prefilter candidate sizes (not for measurement!)
    pxmm0 = float(np.median([m.px_per_mm for m in markers]))
    min_long_px = min_length_mm * pxmm0 * 0.60
    max_long_px = max_length_mm * pxmm0 * 1.60

    marker_boxes = [cv2.boundingRect(m.corners.astype(np.int32).reshape(-1, 1, 2)) for m in markers]

    best = None
    best_score = -1.0

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 150:
            continue

        rect, aspect, long_side, short_side = rotated_rect_features(c)
        if aspect < 1.7:
            continue
        if long_side < min_long_px or long_side > max_long_px:
            continue

        (cx, cy), _, _ = rect

        # Exclude marker regions
        inside_marker = False
        for mx, my, mw, mh in marker_boxes:
            if (mx - 10) <= cx <= (mx + mw + 10) and (my - 10) <= cy <= (my + mh + 10):
                inside_marker = True
                break
        if inside_marker:
            continue

        mask = np.zeros((H, W), np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                iterations=2)

        score = math.sqrt(area) * aspect
        if score > best_score:
            best_score = score
            best = (c, mask, (cx, cy))

    return best


# ----------------------------
# Measurement in mm-space (homography)
# ----------------------------

def median_filter_1d(x: np.ndarray, k: int = 5) -> np.ndarray:
    if x.size < k:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x, dtype=float)
    for i in range(x.size):
        out[i] = float(np.median(xp[i:i + k]))
    return out


def binned_widths(t: np.ndarray, s: np.ndarray, nbins: int = 220, min_pts: int = 10):
    tmin, tmax = float(np.min(t)), float(np.max(t))
    edges = np.linspace(tmin, tmax, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.full(nbins, np.nan, dtype=float)
    for i in range(nbins):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if int(np.count_nonzero(m)) < int(min_pts):
            continue
        ss = s[m]
        widths[i] = float(ss.max() - ss.min())
    return centers, widths


def metric_guess(d_mm: float):
    sizes = [1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    best = min(sizes, key=lambda s: abs(d_mm - s))
    return float(best), float(d_mm - best)


def measure_from_mask_homography(mask: np.ndarray,
                                 H_pix2mm: np.ndarray,
                                 px_per_mm_local: float,
                                 erode_mm: float,
                                 uh_margin_ratio: float,
                                 uh_margin_min: float,
                                 uh_margin_max: float) -> Optional[Dict[str, float]]:
    # Erode in pixels (derived from marker scale) to reduce shadow/halo
    erode_px = int(max(0, round(erode_mm * px_per_mm_local)))
    m = mask.copy()
    if erode_px > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=erode_px)

    ys, xs = np.where(m > 0)
    if xs.size < 800:
        return None

    pts = np.column_stack([xs, ys]).astype(np.float32).reshape(-1, 1, 2)
    pts_mm = cv2.perspectiveTransform(pts, H_pix2mm).reshape(-1, 2)

    mean = pts_mm.mean(axis=0)
    c = pts_mm - mean
    cov = np.cov(c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    minor = eigvecs[:, order[1]]

    t = c.dot(major)
    s = c.dot(minor)

    total_len = float(t.max() - t.min())

    centers, widths = binned_widths(t, s, nbins=240, min_pts=12)
    valid = ~np.isnan(widths)
    if int(np.count_nonzero(valid)) < 60:
        return None

    tn = (centers - float(t.min())) / (total_len + 1e-9)
    vtn = tn[valid]
    vwid = median_filter_1d(widths[valid], 5)

    # Head side decision
    left_mean = float(np.mean(vwid[vtn < 0.12])) if np.any(vtn < 0.12) else float(np.mean(vwid[:10]))
    right_mean = float(np.mean(vwid[vtn > 0.88])) if np.any(vtn > 0.88) else float(np.mean(vwid[-10:]))
    head_on_right = bool(right_mean > left_mean)

    # Gauge from central shaft widths (robust against head/fillet): use a low-mid percentile
    core = vwid[(vtn > 0.30) & (vtn < 0.70)]
    if core.size < 25:
        core = vwid[(vtn > 0.25) & (vtn < 0.75)]
    if core.size < 25:
        core = vwid

    shaft_d = float(np.percentile(core, 30))

    # Underhead boundary threshold
    margin = max(uh_margin_min, uh_margin_ratio * shaft_d)
    margin = min(margin, uh_margin_max)
    thr = shaft_d + margin

    # Find boundary: starting at head end, walk toward tip until width stays below threshold
    idx = np.argsort(vtn)
    tn_s = vtn[idx]
    w_s = vwid[idx]

    boundary = None
    run = 0
    if head_on_right:
        for tval, wval in zip(tn_s[::-1], w_s[::-1]):
            if wval < thr:
                run += 1
                if run >= 6:
                    boundary = float(tval)
                    break
            else:
                run = 0
        under_len = total_len * (boundary if boundary is not None else 1.0)
    else:
        for tval, wval in zip(tn_s, w_s):
            if wval < thr:
                run += 1
                if run >= 6:
                    boundary = float(tval)
                    break
            else:
                run = 0
        under_len = total_len * ((1.0 - boundary) if boundary is not None else 1.0)

    return {
        "shaft_d_mm": shaft_d,
        "total_mm": total_len,
        "under_mm": float(under_len),
        "head_ratio": float(max(left_mean, right_mean) / (np.median(core) + 1e-9)),
        "boundary_frac": float(boundary) if boundary is not None else 1.0,
        "pxmm_used": float(px_per_mm_local),
        "erode_px": float(erode_px),
    }


def put_text(img, text, org, scale=0.9, color=(0, 0, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str)
    src.add_argument("--url", type=str)

    ap.add_argument("--marker-mm", type=float, default=40.0)
    ap.add_argument("--min-length-mm", type=float, default=10.0)
    ap.add_argument("--max-length-mm", type=float, default=70.0)
    ap.add_argument("--length-step", type=float, default=5.0)

    ap.add_argument("--erode-mm", type=float, default=0.9)

    ap.add_argument("--uh-margin-ratio", type=float, default=0.10)
    ap.add_argument("--uh-margin-min", type=float, default=0.30)
    ap.add_argument("--uh-margin-max", type=float, default=0.90)

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save", type=str, default="")

    args = ap.parse_args()

    bgr = imread_url(args.url) if args.url else cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit("Failed to load image")

    markers, marker_th = detect_square_markers(bgr, marker_mm=args.marker_mm)
    if len(markers) < 1:
        raise SystemExit("No marker detected")

    edges, blob = preprocess_blob(bgr)

    best = pick_best_screw_contour(
        bgr=bgr,
        blob=blob,
        markers=markers,
        min_length_mm=args.min_length_mm,
        max_length_mm=args.max_length_mm,
        marker_mm=args.marker_mm
    )
    if best is None:
        raise SystemExit("Failed to segment screw (candidate selection)")

    contour, mask, (cx, cy) = best

    # Use nearest marker for homography
    nearest = min(markers, key=lambda m: (m.center[0] - cx) ** 2 + (m.center[1] - cy) ** 2)
    H_pix2mm = homography_from_marker(nearest, args.marker_mm)
    pxmm_local = nearest.px_per_mm

    meas = measure_from_mask_homography(
        mask=mask,
        H_pix2mm=H_pix2mm,
        px_per_mm_local=pxmm_local,
        erode_mm=args.erode_mm,
        uh_margin_ratio=args.uh_margin_ratio,
        uh_margin_min=args.uh_margin_min,
        uh_margin_max=args.uh_margin_max
    )
    if meas is None:
        raise SystemExit("Measurement failed (mask too small after erosion or unstable widths)")

    m_size, m_err = metric_guess(meas["shaft_d_mm"])
    under_snap = args.length_step * round(meas["under_mm"] / args.length_step)

    annotated = bgr.copy()

    # Draw markers
    for i, m in enumerate(markers[:2]):
        pts = m.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
        put_text(annotated, f"marker{i+1}", (int(m.center[0]), int(m.center[1])), scale=0.7, color=(0, 255, 0), thickness=2)

    cv2.drawContours(annotated, [contour], -1, (255, 0, 0), 2)

    y = 40
    put_text(annotated, f"Result: M{m_size:.0f} x {under_snap:.0f} (step={args.length_step:.0f}mm)", (20, y), scale=1.0); y += 34
    put_text(annotated, f"Gauge: M{m_size:.0f} (shaft ~ {meas['shaft_d_mm']:.2f} mm, err {m_err:+.2f})", (20, y), scale=1.0); y += 34
    put_text(annotated, f"Length (tip->underhead): {meas['under_mm']:.2f} mm (nearest {under_snap:.0f})", (20, y), scale=1.0); y += 34
    put_text(annotated, f"Total length (end-to-end): {meas['total_mm']:.2f} mm", (20, y), scale=1.0); y += 34

    if args.debug:
        put_text(annotated,
                 f"pxmm={meas['pxmm_used']:.3f} erode_px={meas['erode_px']:.0f} boundary={meas['boundary_frac']:.3f}",
                 (20, y), scale=0.8)

    print(f"M{m_size:.0f} x {under_snap:.0f}")
    print(f"shaft_d={meas['shaft_d_mm']:.2f}mm  under={meas['under_mm']:.2f}mm  total={meas['total_mm']:.2f}mm")

    if args.save:
        cv2.imwrite(args.save, annotated)
        print(f"Saved: {args.save}")

    if args.show:
        cv2.imshow("input", bgr)
        cv2.imshow("marker_thresh", marker_th)
        cv2.imshow("edges", edges)
        cv2.imshow("blob", blob)
        cv2.imshow("screw_mask", mask)
        cv2.imshow("annotated", annotated)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
