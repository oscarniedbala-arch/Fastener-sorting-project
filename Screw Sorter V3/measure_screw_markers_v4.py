import argparse
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Utils
# ----------------------------

def imread_url(url: str) -> np.ndarray:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        arr = np.frombuffer(resp.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image from URL")
    return img


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def median_filter_1d(x: np.ndarray, k: int = 5) -> np.ndarray:
    if x.size < k:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x, dtype=float)
    for i in range(x.size):
        out[i] = float(np.median(xp[i:i + k]))
    return out


def binned_widths(t: np.ndarray, s: np.ndarray, tmin: float, tmax: float, nbins: int = 240, min_pts: int = 12):
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


def parse_float_list(s: str) -> List[float]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


# ----------------------------
# Marker detection (outer black square)
# ----------------------------

@dataclass
class Marker:
    corners: np.ndarray  # (4,2) tl,tr,br,bl
    area: float
    center: Tuple[float, float]
    side_px: float
    px_per_mm: float


def detect_square_markers(bgr: np.ndarray, marker_mm: float,
                          min_area_px: int = 2500,
                          close_kernel: int = 9,
                          max_rect_ratio: float = 1.35):
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
        markers.append(Marker(corners=corners, area=area, center=(cx, cy), side_px=side_px, px_per_mm=px_per_mm))

    markers.sort(key=lambda m: m.area, reverse=True)
    return markers, th


def homography_from_marker(marker: Marker, marker_mm: float) -> np.ndarray:
    src = marker.corners.astype(np.float32)
    dst = np.array([[0, 0], [marker_mm, 0], [marker_mm, marker_mm], [0, marker_mm]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


# ----------------------------
# Preprocess + robust screw candidate selection
# ----------------------------

def preprocess_edges_blob(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    v = float(np.median(g))
    lower = int(max(0, 0.55 * v))
    upper = int(min(255, 1.45 * v))
    edges = cv2.Canny(g, lower, upper)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blob = cv2.dilate(edges, k, iterations=2)
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, k, iterations=2)
    return edges, blob


def marker_region_mask(shape_hw, markers: List[Marker], expand_px: int = 12) -> np.ndarray:
    H, W = shape_hw
    m = np.zeros((H, W), np.uint8)
    for mk in markers:
        x, y, w, h = cv2.boundingRect(mk.corners.astype(np.int32).reshape(-1, 1, 2))
        x0 = max(0, x - expand_px)
        y0 = max(0, y - expand_px)
        x1 = min(W, x + w + expand_px)
        y1 = min(H, y + h + expand_px)
        m[y0:y1, x0:x1] = 255
    return m


def markers_roi(shape_hw, markers: List[Marker], margin_mult: float = 2.0):
    H, W = shape_hw
    if len(markers) < 1:
        return (0, 0, W, H)

    # Use up to 2 largest markers to bound ROI
    use = markers[:2]
    pts = np.vstack([m.corners for m in use])
    x0, y0 = np.min(pts[:, 0]), np.min(pts[:, 1])
    x1, y1 = np.max(pts[:, 0]), np.max(pts[:, 1])

    side_px = float(np.median([m.side_px for m in use]))
    margin = int(round(side_px * margin_mult))

    X0 = int(max(0, x0 - margin))
    Y0 = int(max(0, y0 - margin))
    X1 = int(min(W, x1 + margin))
    Y1 = int(min(H, y1 + margin))
    return (X0, Y0, X1 - X0, Y1 - Y0)


def rotated_rect_aspect(c):
    rect = cv2.minAreaRect(c)
    (_, _), (w, h), _ = rect
    w = max(float(w), 1e-6)
    h = max(float(h), 1e-6)
    long_side = max(w, h)
    short_side = min(w, h)
    return rect, long_side / short_side, long_side


def pick_screw_contour(blob: np.ndarray,
                       marker_mask: np.ndarray,
                       roi: Tuple[int, int, int, int],
                       pxmm_hint: float,
                       min_len_mm: float,
                       max_len_mm: float,
                       max_marker_overlap: float = 0.02):
    H, W = blob.shape
    x, y, w, h = roi

    blob2 = blob.copy()

    # Hard-zero marker regions (prevents merges)
    blob2[marker_mask > 0] = 0

    # Hard-zero outside ROI (prevents page border candidates)
    roi_mask = np.zeros_like(blob2)
    roi_mask[y:y + h, x:x + w] = 255
    blob2[roi_mask == 0] = 0

    contours, _ = cv2.findContours(blob2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, blob2

    min_long_px = min_len_mm * pxmm_hint * 0.60
    max_long_px = max_len_mm * pxmm_hint * 1.60

    best = None
    best_score = -1.0

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 150:
            continue

        rect, aspect, long_side = rotated_rect_aspect(c)
        if aspect < 1.7:
            continue
        if long_side < min_long_px or long_side > max_long_px:
            continue

        # Reject anything touching image border too much
        bx, by, bw, bh = cv2.boundingRect(c)
        if bx <= 1 or by <= 1 or (bx + bw) >= (W - 2) or (by + bh) >= (H - 2):
            if long_side > 100:
                continue

        # Build contour mask for overlap test
        cm = np.zeros((H, W), np.uint8)
        cv2.drawContours(cm, [c], -1, 255, -1)

        overlap = float(np.count_nonzero((cm > 0) & (marker_mask > 0))) / max(1.0, float(np.count_nonzero(cm > 0)))
        if overlap > max_marker_overlap:
            continue

        score = (area ** 0.5) * aspect
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        return None, blob2

    # Final filled mask
    screw_mask = np.zeros((H, W), np.uint8)
    cv2.drawContours(screw_mask, [best], -1, 255, -1)
    screw_mask = cv2.morphologyEx(
        screw_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=2
    )
    return (best, screw_mask), blob2


# ----------------------------
# Measurement (homography + dual erosion)
# ----------------------------

def measure_screw(mask: np.ndarray,
                  H_pix2mm: np.ndarray,
                  pxmm_local: float,
                  erode_len_mm: float,
                  erode_dia_mm: float,
                  uh_margin_ratio: float,
                  uh_margin_min: float,
                  uh_margin_max: float):
    # Erode in pixels
    er_len_px = int(max(0, round(erode_len_mm * pxmm_local)))
    er_dia_px = int(max(0, round(erode_dia_mm * pxmm_local)))

    m_len = mask.copy()
    m_dia = mask.copy()

    if er_len_px > 0:
        m_len = cv2.erode(m_len, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=er_len_px)
    if er_dia_px > 0:
        m_dia = cv2.erode(m_dia, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=er_dia_px)

    ys, xs = np.where(m_len > 0)
    if xs.size < 800:
        return None
    pts_len = np.column_stack([xs, ys]).astype(np.float32).reshape(-1, 1, 2)
    mm_len = cv2.perspectiveTransform(pts_len, H_pix2mm).reshape(-1, 2)

    # PCA axis from length mask
    mean = mm_len.mean(axis=0)
    c = mm_len - mean
    cov = np.cov(c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    minor = eigvecs[:, order[1]]

    t_len = c.dot(major)
    s_len = c.dot(minor)
    tmin, tmax = float(t_len.min()), float(t_len.max())
    total_len = float(tmax - tmin)

    # Diameter mask points projected onto the same axis system
    ys2, xs2 = np.where(m_dia > 0)
    if xs2.size < 800:
        return None
    pts_d = np.column_stack([xs2, ys2]).astype(np.float32).reshape(-1, 1, 2)
    mm_d = cv2.perspectiveTransform(pts_d, H_pix2mm).reshape(-1, 2)
    cd = mm_d - mean
    t_d = cd.dot(major)
    s_d = cd.dot(minor)

    # Width profile for diameter from the more-eroded mask (removes halo)
    centers_d, widths_d = binned_widths(t_d, s_d, tmin, tmax, nbins=240, min_pts=14)
    valid_d = ~np.isnan(widths_d)
    if int(np.count_nonzero(valid_d)) < 60:
        return None
    tn = (centers_d - tmin) / (total_len + 1e-9)
    vtn = tn[valid_d]
    vwid_d = median_filter_1d(widths_d[valid_d], 5)

    core = vwid_d[(vtn > 0.30) & (vtn < 0.70)]
    if core.size < 20:
        core = vwid_d[(vtn > 0.25) & (vtn < 0.75)]
    if core.size < 20:
        core = vwid_d

    # Use a conservative percentile so we don't get pushed up by residual shadows
    shaft_d = float(np.percentile(core, 25))

    # Width profile for head boundary from the less-eroded mask (keeps head shape)
    centers_l, widths_l = binned_widths(t_len, s_len, tmin, tmax, nbins=240, min_pts=14)
    valid_l = ~np.isnan(widths_l)
    if int(np.count_nonzero(valid_l)) < 60:
        return None
    tn_l = (centers_l[valid_l] - tmin) / (total_len + 1e-9)
    w_l = median_filter_1d(widths_l[valid_l], 5)

    left_mean = float(np.mean(w_l[tn_l < 0.12])) if np.any(tn_l < 0.12) else float(np.mean(w_l[:10]))
    right_mean = float(np.mean(w_l[tn_l > 0.88])) if np.any(tn_l > 0.88) else float(np.mean(w_l[-10:]))
    head_on_right = bool(right_mean > left_mean)

    margin = max(uh_margin_min, uh_margin_ratio * shaft_d)
    margin = min(margin, uh_margin_max)
    thr = shaft_d + margin

    # Boundary search on the less-eroded profile
    idx = np.argsort(tn_l)
    tn_s = tn_l[idx]
    w_s = w_l[idx]

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
        "pxmm": float(pxmm_local),
        "er_len_px": int(er_len_px),
        "er_dia_px": int(er_dia_px),
        "boundary": float(boundary) if boundary is not None else 1.0,
    }


def snap(x: float, step: float) -> float:
    if step <= 0:
        return x
    return float(np.round(x / step) * step)


def nearest_metric(d_mm: float, gauges: List[float]) -> Tuple[float, float]:
    best = min(gauges, key=lambda g: abs(d_mm - g))
    return float(best), float(d_mm - best)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str)
    src.add_argument("--url", type=str)

    ap.add_argument("--marker-mm", type=float, default=40.0)
    ap.add_argument("--min-length-mm", type=float, default=10.0)
    ap.add_argument("--max-length-mm", type=float, default=70.0)
    ap.add_argument("--length-step", type=float, default=5.0)

    # Dual erosion (this is the key change)
    ap.add_argument("--erode-len-mm", type=float, default=0.8, help="Erosion used for length (mm)")
    ap.add_argument("--erode-dia-mm", type=float, default=1.6, help="Erosion used for diameter (mm)")

    ap.add_argument("--uh-margin-ratio", type=float, default=0.10)
    ap.add_argument("--uh-margin-min", type=float, default=0.30)
    ap.add_argument("--uh-margin-max", type=float, default=0.90)

    # If you do not own M8 screws, do not allow M8 in the classifier.
    ap.add_argument("--gauges", type=str, default="2,2.5,3,4,5,6", help="Comma list of allowed metric sizes")

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--save", type=str, default="")

    args = ap.parse_args()
    gauges = parse_float_list(args.gauges)

    bgr = imread_url(args.url) if args.url else cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit("Failed to load image")

    markers, marker_th = detect_square_markers(bgr, marker_mm=args.marker_mm)
    if len(markers) < 1:
        raise SystemExit("No marker detected")

    edges, blob = preprocess_edges_blob(bgr)

    # ROI + marker mask to prevent catastrophic merges
    roi = markers_roi(blob.shape, markers, margin_mult=2.0)
    mkmask = marker_region_mask(blob.shape, markers, expand_px=14)

    pxmm_hint = float(np.median([m.px_per_mm for m in markers]))

    picked, blob2 = pick_screw_contour(
        blob=blob,
        marker_mask=mkmask,
        roi=roi,
        pxmm_hint=pxmm_hint,
        min_len_mm=args.min_length_mm,
        max_len_mm=args.max_length_mm,
        max_marker_overlap=0.01
    )
    if picked is None:
        raise SystemExit("Failed to segment screw (try adjusting lighting or move screw away from markers)")

    contour, screw_mask = picked

    # Screw center for nearest marker selection
    bx, by, bw, bh = cv2.boundingRect(contour)
    scx, scy = bx + bw / 2.0, by + bh / 2.0
    nearest = min(markers, key=lambda m: (m.center[0] - scx) ** 2 + (m.center[1] - scy) ** 2)

    H_pix2mm = homography_from_marker(nearest, args.marker_mm)

    meas = measure_screw(
        mask=screw_mask,
        H_pix2mm=H_pix2mm,
        pxmm_local=nearest.px_per_mm,
        erode_len_mm=args.erode_len_mm,
        erode_dia_mm=args.erode_dia_mm,
        uh_margin_ratio=args.uh_margin_ratio,
        uh_margin_min=args.uh_margin_min,
        uh_margin_max=args.uh_margin_max,
    )
    if meas is None:
        raise SystemExit("Measurement failed (mask too small after erosion or unstable profile)")

    m_size, m_err = nearest_metric(meas["shaft_d_mm"], gauges)
    under_snap = snap(meas["under_mm"], args.length_step)

    annotated = bgr.copy()

    # Draw markers
    for i, mk in enumerate(markers[:2]):
        pts = mk.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
        cv2.putText(annotated, f"marker{i+1}", (int(mk.center[0]), int(mk.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.drawContours(annotated, [contour], -1, (255, 0, 0), 2)

    y = 40
    def put(line):
        nonlocal y
        cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        y += 34

    put(f"Result: M{m_size:.0f} x {under_snap:.0f} (step={args.length_step:.0f}mm)")
    put(f"Gauge: M{m_size:.0f} (shaft ~ {meas['shaft_d_mm']:.2f} mm, err {m_err:+.2f})")
    put(f"Length (tip->underhead): {meas['under_mm']:.2f} mm (nearest {under_snap:.0f})")
    put(f"Total length (end-to-end): {meas['total_mm']:.2f} mm")

    if args.debug:
        put(f"pxmm={meas['pxmm']:.3f} er_len_px={meas['er_len_px']} er_dia_px={meas['er_dia_px']} boundary={meas['boundary']:.3f}")

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
        cv2.imshow("blob_masked", blob2)
        cv2.imshow("screw_mask", screw_mask)
        cv2.imshow("annotated", annotated)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
