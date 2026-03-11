# This script captures images from an ESP32-CAM, preprocesses them to isolate screws,
# and uploads them to Edge Impulse for machine learning purposes.

#OUTDATED - DONT USE THIS!


import argparse
import os
import time
from datetime import datetime

import requests
import edgeimpulse as ei

import cv2
import numpy as np


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def capture_jpeg(url: str, timeout_s: int = 10) -> bytes:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.content


def decode_jpeg_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("JPEG decode failed")
    return img


def apply_flip(img: np.ndarray, flip: str) -> np.ndarray:
    if flip == "none":
        return img
    if flip == "h":
        return cv2.flip(img, 1)
    if flip == "v":
        return cv2.flip(img, 0)
    if flip == "hv":
        return cv2.flip(img, -1)
    raise ValueError(f"Unknown flip mode: {flip}")


def estimate_px_per_mm_from_squares(bgr: np.ndarray, square_mm: float = 5.0) -> float | None:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # White-ish squares: low saturation, high value (tune if needed)
    mask = cv2.inRange(hsv, (0, 0, 160), (180, 90, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    widths = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 80 or area > 30000:
            continue
        ar = w / float(h + 1e-6)
        if ar < 0.65 or ar > 1.55:
            continue
        widths.append((w + h) / 2.0)

    if not widths:
        return None

    square_px = float(np.median(widths))
    return square_px / square_mm


def find_screw_bbox(bgr: np.ndarray, min_rot_ar: float = 1.4) -> tuple[int, int, int, int] | None:
    """
    Returns bbox (x,y,w,h) for best candidate screw contour.
    Uses a rotation-invariant aspect ratio gate to reject grid blobs.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 30, 120)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(bgr.shape[0] * bgr.shape[1])

    best = None
    best_score = -1.0

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 350.0:
            continue

        # Rotation-invariant elongation
        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), _ = rect
        if rw < 5 or rh < 5:
            continue
        rot_ar = max(rw, rh) / float(min(rw, rh) + 1e-6)
        if rot_ar < min_rot_ar:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # Reject absurdly huge boxes (often merged background/hand)
        if (w * h) > 0.45 * frame_area:
            continue

        # Score prefers large + elongated
        score = area * min(rot_ar, 8.0)
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    return best


def crop_and_scale_normalize(
    bgr: np.ndarray,
    out_size: int = 128,
    px_per_mm: float | None = None,
    target_px_per_mm: float | None = None,
    pad_frac: float = 0.8,
    reject_edge: bool = False,
    edge_margin: int = 12,
    min_rot_ar: float = 1.4,
) -> np.ndarray | None:
    img = bgr

    if target_px_per_mm is not None and px_per_mm is not None and px_per_mm > 1e-6:
        scale = float(np.clip(target_px_per_mm / px_per_mm, 0.5, 2.0))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    bbox = find_screw_bbox(img, min_rot_ar=min_rot_ar)
    if bbox is None:
        return None

    x, y, w, h = bbox

    if reject_edge:
        if (x < edge_margin) or (y < edge_margin) or (x + w > img.shape[1] - edge_margin) or (y + h > img.shape[0] - edge_margin):
            return None

    side = int(max(w, h) * (1.0 + pad_frac))
    cx, cy = x + w // 2, y + h // 2

    x0 = max(cx - side // 2, 0)
    y0 = max(cy - side // 2, 0)
    x1 = min(x0 + side, img.shape[1])
    y1 = min(y0 + side, img.shape[0])

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def encode_jpeg(bgr: np.ndarray, quality: int = 95) -> bytes:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encode failed")
    return enc.tobytes()


def debug_overlay(bgr: np.ndarray, px_per_mm: float | None, min_rot_ar: float) -> np.ndarray:
    img = bgr.copy()
    bbox = find_screw_bbox(img, min_rot_ar=min_rot_ar)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    txt = f"px/mm: {px_per_mm:.2f}" if px_per_mm else "px/mm: (none)"
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def crop_has_skin(bgr_crop: np.ndarray, frac_thresh: float = 0.03) -> bool:
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 20, 70), (25, 255, 255))
    m2 = cv2.inRange(hsv, (160, 20, 70), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    frac = float(np.count_nonzero(mask)) / float(mask.size)
    return frac > frac_thresh


def dark_p1(gray_bgr_crop: np.ndarray, p: float = 1.0) -> float:
    """
    1st percentile of grayscale (lower => more dark pixels present).
    Grid-only crops tend to have higher p1; screw crops tend to have lower p1.
    """
    gray = cv2.cvtColor(gray_bgr_crop, cv2.COLOR_BGR2GRAY)
    return float(np.percentile(gray, p))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--esp32", required=True, help="e.g. http://192.168.1.50")
    p.add_argument("--endpoint", default="/capture.jpg")
    p.add_argument("--label", required=True, help="e.g. M4, M5, M6")
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--interval", type=float, default=3.0)
    p.add_argument("--category", default="training", choices=["training", "testing"])
    p.add_argument("--api-key", default=os.environ.get("EI_API_KEY"))
    p.add_argument("--sessions-root", default="sessions")

    # Preprocess
    p.add_argument("--out-size", type=int, default=128)
    p.add_argument("--flip", default="none", choices=["none", "h", "v", "hv"])
    p.add_argument("--target-px-per-mm", type=float, default=0.0, help="Scale-normalize target px/mm. Set 0 to disable.")
    p.add_argument("--pad-frac", type=float, default=0.8, help="Padding around screw bbox (0.6–1.0 typical).")
    p.add_argument("--jpeg-quality", type=int, default=95)

    # False-positive controls
    p.add_argument("--min-rot-ar", type=float, default=1.4, help="Reject non-elongated contours (grid blobs). 1.3–1.8 typical.")
    p.add_argument("--reject-empty", action="store_true", help="Reject crops that look grid-only (no screw).")
    p.add_argument("--dark-p1-max", type=float, default=75.0, help="Reject if 1st percentile gray > this (grid-only tends to be high).")

    # Quality controls
    p.add_argument("--reject-skin", action="store_true", help="Skip frames if skin is detected in the final crop.")
    p.add_argument("--skin-frac", type=float, default=0.03, help="Skin fraction threshold (0.02–0.05 typical).")
    p.add_argument("--reject-edge", action="store_true", help="Skip frames where detected bbox is near image edge.")
    p.add_argument("--edge-margin", type=int, default=12, help="Edge margin in pixels for reject-edge.")

    # Debug
    p.add_argument("--debug-every", type=int, default=0, help="Save overlay image every N frames (0 disables).")

    # Upload control
    p.add_argument("--no-upload", action="store_true", help="Do not upload to Edge Impulse (save locally only).")

    args = p.parse_args()

    session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(args.sessions_root, session_id, args.label)
    ensure_dir(out_dir)

    debug_dir = os.path.join(args.sessions_root, session_id, "_debug")
    if args.debug_every:
        ensure_dir(debug_dir)

    base = args.esp32.rstrip("/")
    url = base + args.endpoint

    print(f"Session: {session_id}")
    print(f"Capturing from: {url}")
    print(f"Saving to: {out_dir}")
    if args.debug_every:
        print(f"Debug overlays: {debug_dir} (NOT uploaded)")
    print(
        f"Preprocess: out={args.out_size} flip={args.flip} target_px_per_mm={args.target_px_per_mm} pad_frac={args.pad_frac}\n"
        f"Reject: edge={args.reject_edge} skin={args.reject_skin} empty={args.reject_empty} min_rot_ar={args.min_rot_ar}"
    )

    saved = 0
    skipped = 0

    for i in range(args.count):
        try:
            jpeg = capture_jpeg(url)
            bgr = decode_jpeg_to_bgr(jpeg)
            bgr = apply_flip(bgr, args.flip)

            px_per_mm = estimate_px_per_mm_from_squares(bgr, square_mm=5.0)
            target = args.target_px_per_mm if args.target_px_per_mm > 0 else None

            crop = crop_and_scale_normalize(
                bgr,
                out_size=args.out_size,
                px_per_mm=px_per_mm,
                target_px_per_mm=target,
                pad_frac=args.pad_frac,
                reject_edge=args.reject_edge,
                edge_margin=args.edge_margin,
                min_rot_ar=args.min_rot_ar,
            )

            if crop is None:
                skipped += 1
                print(f"[{i+1}/{args.count}] skipped (no bbox / edge reject / ar reject)")
                time.sleep(args.interval)
                continue

            if args.reject_skin and crop_has_skin(crop, frac_thresh=args.skin_frac):
                skipped += 1
                print(f"[{i+1}/{args.count}] skipped (skin detected)")
                time.sleep(args.interval)
                continue

            if args.reject_empty:
                p1 = dark_p1(crop, p=1.0)
                if p1 > args.dark_p1_max:
                    skipped += 1
                    print(f"[{i+1}/{args.count}] skipped (empty/grid-only suspected; p1={p1:.1f})")
                    time.sleep(args.interval)
                    continue

            out_jpeg = encode_jpeg(crop, quality=args.jpeg_quality)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(out_dir, f"{args.label}.{ts}.jpg")
            with open(path, "wb") as f:
                f.write(out_jpeg)

            if args.debug_every and ((i + 1) % args.debug_every == 0):
                dbg = debug_overlay(bgr, px_per_mm, min_rot_ar=args.min_rot_ar)
                dbg_path = os.path.join(debug_dir, f"DEBUG.{args.label}.{ts}.png")
                cv2.imwrite(dbg_path, dbg)

            meta = f"px/mm={px_per_mm:.2f}" if px_per_mm else "px/mm=none"
            saved += 1
            print(f"[{i+1}/{args.count}] saved {os.path.basename(path)} ({meta})")

        except Exception as e:
            skipped += 1
            print(f"[{i+1}/{args.count}] capture failed: {e}")

        time.sleep(args.interval)

    print(f"Capture complete: saved={saved}, skipped={skipped}")

    if args.no_upload:
        print("No-upload enabled; skipping Edge Impulse upload.")
        return

    if not args.api_key:
        raise SystemExit("Missing API key. Provide --api-key or set EI_API_KEY, or use --no-upload.")

    ei.API_KEY = args.api_key
    print("Uploading session folder to Edge Impulse...")

    resp = ei.experimental.data.upload_directory(
        directory=out_dir,
        category=args.category,
        label=args.label,
        metadata={"source": "esp32-cam", "session": session_id},
    )

    if len(resp.fails) > 0:
        print("Upload failures:")
        for f in resp.fails:
            print(f"  - {f}")
        raise SystemExit(1)

    print(f"Uploaded {len(resp.successes)} images for label={args.label} ({args.category}).")


if __name__ == "__main__":
    main()
