import argparse
import math
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np


# ----------------------------
# Utility
# ----------------------------

def imread_url(url: str) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
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


def snap_to_step(v_mm: float, step_mm: float) -> float:
    if step_mm <= 0:
        return v_mm
    return float(step_mm * round(v_mm / step_mm))


def metric_guess(d_mm: float):
    # You can extend this list if you need M1.2/M1.4 etc.
    sizes = [1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    best = min(sizes, key=lambda s: abs(d_mm - s))
    err = d_mm - best
    return best, err


def put_text(img, text, org, scale=0.9, color=(0, 0, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ----------------------------
# Marker detection (square markers)
# ----------------------------

@dataclass
class Marker:
    corners: np.ndarray
    area: float
    center: tuple
    side_px: float
    px_per_mm: float


def detect_square_markers(bgr: np.ndarray,
                          marker_mm: float,
                          min_area_px: int = 2500,
                          close_kernel: int = 9,
                          max_rect_ratio: float = 1.30):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = []
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
        rect_ratio = max(sides) / max(min(sides), 1e-6)
        if rect_ratio > max_rect_ratio:
            continue

        side_px = float(np.mean(sides))
        cx, cy = float(corners[:, 0].mean()), float(corners[:, 1].mean())
        px_per_mm = side_px / float(marker_mm)

        markers.append(Marker(corners=corners, area=area, center=(cx, cy), side_px=side_px, px_per_mm=px_per_mm))

    markers.sort(key=lambda m: m.area, reverse=True)
    return markers, th, closed


def px_per_mm_at(markers, pt_xy, marker_mm: float):
    # Weighted by inverse distance to reduce distortion effects a bit.
    if not markers:
        return None
    x, y = pt_xy
    vals = []
    wts = []
    for m in markers:
        cx, cy = m.center
        d = math.hypot(cx - x, cy - y)
        w = 1.0 / max(d, 20.0)  # cap weight
        vals.append(m.px_per_mm)
        wts.append(w)
    return float(np.dot(vals, wts) / np.sum(wts))


# ----------------------------
# Screw segmentation + candidate selection
# ----------------------------

def preprocess_edges(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    v = float(np.median(g))
    lower = int(max(0, 0.50 * v))
    upper = int(min(255, 1.50 * v))
    edges = cv2.Canny(g, lower, upper)

    gx = cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(
        cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)
    )
    thr_val = float(np.percentile(mag, 92))
    _, mag_th = cv2.threshold(mag, thr_val, 255, cv2.THRESH_BINARY)

    comb = cv2.bitwise_or(edges, mag_th)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blob = cv2.dilate(comb, k, iterations=2)
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, k, iterations=2)
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, k, iterations=1)

    return edges, mag_th, blob


def rotated_rect_features(contour):
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), ang = rect
    w = max(float(w), 1e-6)
    h = max(float(h), 1e-6)
    long_side = max(w, h)
    short_side = min(w, h)
    aspect = long_side / short_side
    return rect, aspect, long_side, short_side


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


def pca_axes_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    c = pts - mean
    cov = np.cov(c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    minor = eigvecs[:, order[1]]
    t = c.dot(major)
    s = c.dot(minor)
    return mean, major, minor, t, s


def measure_length_underhead_from_mask(mask: np.ndarray,
                                      px_per_mm: float,
                                      erode_mm: float,
                                      uh_margin_mm: float,
                                      head_ratio_min: float):
    # Returns total_mm, underhead_mm, plus head metadata for later diameter estimation
    erode_px = max(1, int(round(erode_mm * px_per_mm)))
    m = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=erode_px)

    mean, major, minor, t, s = pca_axes_from_mask(m)
    total_len_px = float(t.max() - t.min())
    total_mm = total_len_px / px_per_mm

    centers, widths = binned_widths(t, s, nbins=220, min_pts=10)
    valid = ~np.isnan(widths)
    if int(np.count_nonzero(valid)) < 40:
        return None

    tn = (centers - float(t.min())) / (total_len_px + 1e-9)
    vtn = tn[valid]
    vwid = widths[valid]
    vwid = median_filter_1d(vwid, 5)

    core = vwid[(vtn > 0.15) & (vtn < 0.85)]
    if core.size < 20:
        core = vwid

    shaft_px = float(np.percentile(core, 25))
    mid_ref = float(np.median(core))

    left = float(np.mean(vwid[vtn < 0.12])) if np.any(vtn < 0.12) else float(np.mean(vwid[:10]))
    right = float(np.mean(vwid[vtn > 0.88])) if np.any(vtn > 0.88) else float(np.mean(vwid[-10:]))
    head_on_right = bool(right > left)
    head_ratio = float(max(left, right) / (mid_ref + 1e-9))

    under_len_px = total_len_px
    boundary_frac = 1.0

    if head_ratio >= head_ratio_min:
        thr_px = shaft_px + (uh_margin_mm * px_per_mm)

        idx = np.argsort(vtn)
        tn_s = vtn[idx]
        wid_s = vwid[idx]

        run = 0
        boundary = None

        if head_on_right:
            scan_t = tn_s[::-1]
            scan_w = wid_s[::-1]
            for tval, wval in zip(scan_t, scan_w):
                if wval < thr_px:
                    run += 1
                    if run >= 6:
                        boundary = float(tval)
                        break
                else:
                    run = 0
            if boundary is not None:
                boundary_frac = boundary
                under_len_px = boundary * total_len_px
        else:
            for tval, wval in zip(tn_s, wid_s):
                if wval < thr_px:
                    run += 1
                    if run >= 6:
                        boundary = float(tval)
                        break
                else:
                    run = 0
            if boundary is not None:
                boundary_frac = boundary
                under_len_px = (1.0 - boundary) * total_len_px

    under_mm = under_len_px / px_per_mm
    return {
        "total_mm": total_mm,
        "under_mm": under_mm,
        "head_ratio": head_ratio,
        "head_on_right": head_on_right,
        "boundary_frac": boundary_frac,
        "mask_eroded": m,
        "pca_mean": mean,
        "pca_major": major,
        "pca_minor": minor,
        "t_min": float(t.min()),
        "t_max": float(t.max()),
    }

def shaft_diameter_from_mask(meta, px_per_mm: float) -> float | None:
    # Use the eroded mask silhouette, measure widths along the PCA minor axis,
    # take a robust percentile in the central region (avoids head).
    m = meta["mask_eroded"]
    ys, xs = np.where(m > 0)
    if xs.size < 500:
        return None

    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    c = pts - mean
    cov = np.cov(c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = eigvecs[:, order[0]]
    minor = eigvecs[:, order[1]]

    t = c.dot(major)
    s = c.dot(minor)

    centers, widths = binned_widths(t, s, nbins=220, min_pts=10)
    valid = ~np.isnan(widths)
    if int(np.count_nonzero(valid)) < 40:
        return None

    tn = (centers - float(t.min())) / (float(t.max() - t.min()) + 1e-9)
    vtn = tn[valid]
    vwid = median_filter_1d(widths[valid], 5)

    # Central shaft region
    core = vwid[(vtn > 0.25) & (vtn < 0.75)]
    if core.size < 20:
        core = vwid

    # Use a lower percentile to avoid occasional bulges/shadows
    diam_px = float(np.percentile(core, 35))
    return diam_px / px_per_mm


def shaft_diameter_from_edges(bgr: np.ndarray,
                             edges: np.ndarray,
                             mask: np.ndarray,
                             meta,
                             px_per_mm: float):
    # Robust width from edge pixels (percentile span), inside shaft region.
    # This avoids “fat masks” and stabilizes M-size classification.
    m2 = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    ey, ex = np.where((edges > 0) & (m2 > 0))
    if ex.size < 120:
        return None

    pts = np.column_stack([ex, ey]).astype(np.float32)
    c = pts - meta["pca_mean"]
    t = c.dot(meta["pca_major"])
    s = c.dot(meta["pca_minor"])

    tmin = meta["t_min"]
    tmax = meta["t_max"]
    tn = (t - tmin) / (tmax - tmin + 1e-9)

    # Define a “shaft band” in t:
    # If head boundary exists, exclude the head end; else use central 50%.
    if meta["head_ratio"] >= 1.15 and meta["boundary_frac"] < 0.98:
        if meta["head_on_right"]:
            tn_lo = 0.20
            tn_hi = max(0.30, meta["boundary_frac"] - 0.06)
        else:
            tn_lo = min(0.70, meta["boundary_frac"] + 0.06)
            tn_hi = 0.80
    else:
        tn_lo, tn_hi = 0.25, 0.75

    m = (tn >= tn_lo) & (tn <= tn_hi)
    s2 = s[m]
    if s2.size < 120:
        # fallback: widen band
        m = (tn >= 0.20) & (tn <= 0.80)
        s2 = s[m]
        if s2.size < 120:
            return None

    lo = float(np.percentile(s2, 3))
    hi = float(np.percentile(s2, 97))
    diam_px = hi - lo
    diam_mm = diam_px / px_per_mm
    return diam_mm


def select_best_screw_candidate(bgr: np.ndarray,
                               markers,
                               marker_mm: float,
                               min_length_mm: float,
                               max_length_mm: float,
                               erode_mm: float):
    edges, mag_th, blob = preprocess_edges(bgr)

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, (edges, mag_th, blob)

    pxmm0 = float(np.median([m.px_per_mm for m in markers]))
    min_long_px = min_length_mm * pxmm0 * 0.60
    max_long_px = max_length_mm * pxmm0 * 1.50

    marker_boxes = [cv2.boundingRect(m.corners.astype(np.int32).reshape(-1, 1, 2)) for m in markers]

    candidates = []
    H, W = bgr.shape[:2]

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 150:
            continue

        rect, aspect, long_side, short_side = rotated_rect_features(c)

        if long_side < min_long_px or long_side > max_long_px:
            continue
        if aspect < 1.7:
            continue
        if long_side > 0.95 * max(H, W):
            continue

        (cx, cy), _, _ = rect

        # Exclude marker regions by center test
        inside_marker = False
        for mx, my, mw, mh in marker_boxes:
            if (mx - 10) <= cx <= (mx + mw + 10) and (my - 10) <= cy <= (my + mh + 10):
                inside_marker = True
                break
        if inside_marker:
            continue

        mask = np.zeros((H, W), np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)

        # Local px/mm (better than global) for measurement
        pxmm = px_per_mm_at(markers, (cx, cy), marker_mm)
        if pxmm is None:
            continue

        meta = measure_length_underhead_from_mask(
            mask=mask,
            px_per_mm=pxmm,
            erode_mm=erode_mm,
            uh_margin_mm=0.50,
            head_ratio_min=1.15
        )
        if meta is None:
            continue

        # Prefer candidates that look like a real screw (head_ratio helps, but isn’t mandatory)
        score = (math.sqrt(area) * aspect) * (1.0 + min(meta["head_ratio"], 2.0))
        candidates.append((score, c, mask, rect, meta, pxmm))

    if not candidates:
        return None, (edges, mag_th, blob)

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0], (edges, mag_th, blob)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="", help="Path to image file")
    ap.add_argument("--url", type=str, default="", help="URL to image (e.g. ESP32 capture.jpg)")
    ap.add_argument("--marker-mm", type=float, default=40.0, help="Printed marker side length in mm")
    ap.add_argument("--min-length-mm", type=float, default=6.0, help="Min expected screw length (mm)")
    ap.add_argument("--max-length-mm", type=float, default=80.0, help="Max expected screw length (mm)")
    ap.add_argument("--length-step", type=float, default=5.0, help="Snap reported length to this step (mm)")
    ap.add_argument("--erode-mm", type=float, default=0.9, help="Mask erosion in mm (stabilizes edges vs glare)")
    ap.add_argument("--show", action="store_true", help="Show debug windows")
    ap.add_argument("--debug", action="store_true", help="Extra debug overlays")
    args = ap.parse_args()

    if not args.image and not args.url:
        raise SystemExit("Provide --image or --url")

    if args.url:
        bgr = imread_url(args.url)
    else:
        bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)

    if bgr is None:
        raise SystemExit("Failed to load image")

    annotated = bgr.copy()

    markers, th, closed = detect_square_markers(bgr, marker_mm=args.marker_mm, min_area_px=2500)

    if not markers:
        put_text(annotated, "ERROR: No marker detected", (20, 40), scale=1.0)
        if args.show:
            cv2.imshow("annotated", annotated)
            cv2.waitKey(0)
        else:
            print("ERROR: No marker detected")
        return

    # Draw markers
    for i, m in enumerate(markers[:2]):
        pts = m.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
        put_text(annotated, f"marker{i+1}", (int(m.center[0]), int(m.center[1])), scale=0.7, color=(0, 255, 0), thickness=2)

    best, dbg = select_best_screw_candidate(
        bgr=bgr,
        markers=markers,
        marker_mm=args.marker_mm,
        min_length_mm=args.min_length_mm,
        max_length_mm=args.max_length_mm,
        erode_mm=args.erode_mm
    )
    edges, mag_th, blob = dbg

    if best is None:
        put_text(annotated, "ERROR: Failed to segment screw (candidate selection)", (20, 40), scale=1.0)
        if args.show:
            cv2.imshow("input", bgr)
            cv2.imshow("edges", edges)
            cv2.imshow("blob", blob)
            cv2.imshow("annotated", annotated)
            cv2.waitKey(0)
        else:
            print("ERROR: Failed to segment screw")
        return

    score, contour, mask, rect, meta, pxmm = best

    # Edge-based shaft diameter (robust)
    shaft_mm = shaft_diameter_from_mask(meta, pxmm)

    # Fallback if edges are too weak
    if shaft_mm is None:
        # Use mask-based shaft estimate only as a last resort (can be inflated under glare)
        shaft_mm = float("nan")

    gauge_m, gauge_err = metric_guess(shaft_mm) if math.isfinite(shaft_mm) else (float("nan"), float("nan"))

    total_mm = meta["total_mm"]
    under_mm = meta["under_mm"]

    under_snap = snap_to_step(under_mm, args.length_step)
    result_text = f"M{gauge_m:.0f} x {under_snap:.0f}" if math.isfinite(gauge_m) else f"? x {under_snap:.0f}"

    # Outline screw
    cv2.drawContours(annotated, [contour], -1, (255, 0, 0), 2)

    # Binary mask view
    screw_mask_vis = mask.copy()

    # Text overlay
    y0 = 40
    put_text(annotated, f"Result: {result_text} (step={args.length_step:.0f}mm)", (20, y0), scale=1.0)
    y0 += 34
    if math.isfinite(shaft_mm):
        put_text(annotated, f"Gauge: M{gauge_m:.0f} (shaft ~ {shaft_mm:.2f} mm, err {gauge_err:+.2f})", (20, y0), scale=1.0)
    else:
        put_text(annotated, "Gauge: UNKNOWN (edges too weak)", (20, y0), scale=1.0)
    y0 += 34
    put_text(annotated, f"Length (tip->underhead): {under_mm:.2f} mm (nearest {under_snap:.0f})", (20, y0), scale=1.0)
    y0 += 34
    put_text(annotated, f"Total length (end-to-end): {total_mm:.2f} mm", (20, y0), scale=1.0)

    if args.debug:
        put_text(annotated, f"head_ratio={meta['head_ratio']:.2f} boundary={meta['boundary_frac']:.3f} pxmm={pxmm:.3f}", (20, y0 + 34), scale=0.8)

    if args.show:
        cv2.imshow("input", bgr)
        cv2.imshow("marker_thresh", th)
        cv2.imshow("edges", edges)
        cv2.imshow("blob", blob)
        cv2.imshow("screw_mask", screw_mask_vis)
        cv2.imshow("annotated", annotated)
        cv2.waitKey(0)
    else:
        print(result_text)
        print(f"Gauge: M{gauge_m:.0f} (shaft ~ {shaft_mm:.2f} mm, err {gauge_err:+.2f})" if math.isfinite(shaft_mm) else "Gauge: UNKNOWN")
        print(f"Underhead: {under_mm:.2f} mm (snapped {under_snap:.0f})")
        print(f"Total: {total_mm:.2f} mm")


if __name__ == "__main__":
    main()
