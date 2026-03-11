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
    # flip: none | h | v | hv
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
    """
    Detect the white squares and estimate their median width in pixels.
    Returns px/mm. If detection fails, returns None.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # White-ish squares: low saturation, high value. Tune if needed.
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


def find_screw_bbox(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Returns (x, y, w, h) of best candidate screw blob.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = -1.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 250.0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = max(w, h) / float(min(w, h) + 1e-6)  # elongation
        score = area * min(ar, 6.0)
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    return best


def crop_and_scale_normalize(
    bgr: np.ndarray,
    out_size: int = 128,
    px_per_mm: float | None = None,
    target_px_per_mm: float | None = None,
    pad_frac: float = 0.25,
) -> np.ndarray:
    """
    Optional: scale-normalize so that px/mm matches target (compensates variable height),
    then find screw bbox, crop square with padding, resize to out_size.
    """
    img = bgr

    if target_px_per_mm is not None and px_per_mm is not None and px_per_mm > 1e-6:
        scale = float(np.clip(target_px_per_mm / px_per_mm, 0.5, 2.0))
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    bbox = find_screw_bbox(img)
    if bbox is None:
        # Fallback: center-crop
        h, w = img.shape[:2]
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        crop = img[y0:y0 + s, x0:x0 + s]
        return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    x, y, w, h = bbox
    side = int(max(w, h) * (1.0 + pad_frac))
    cx, cy = x + w // 2, y + h // 2

    x0 = max(cx - side // 2, 0)
    y0 = max(cy - side // 2, 0)
    x1 = min(x0 + side, img.shape[1])
    y1 = min(y0 + side, img.shape[0])

    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def encode_jpeg(bgr: np.ndarray, quality: int = 95) -> bytes:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encode failed")
    return enc.tobytes()


def debug_overlay(bgr: np.ndarray, px_per_mm: float | None) -> np.ndarray:
    img = bgr.copy()
    bbox = find_screw_bbox(img)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    txt = f"px/mm: {px_per_mm:.2f}" if px_per_mm else "px/mm: (none)"
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    return img


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
    p.add_argument("--pad-frac", type=float, default=0.8, help="Crop padding around detected screw bbox (0.6–1.0 typical).")

    # Preprocess options
    p.add_argument("--out-size", type=int, default=128)
    p.add_argument("--flip", default="none", choices=["none", "h", "v", "hv"])
    p.add_argument("--target-px-per-mm", type=float, default=10.0, help="Scale-normalize target. Set 0 to disable.")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--debug-every", type=int, default=0, help="Save overlay image every N frames (0 disables).")
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Provide --api-key or set EI_API_KEY.")

    session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(args.sessions_root, session_id, args.label)
    ensure_dir(out_dir)

    base = args.esp32.rstrip("/")
    url = base + args.endpoint

    print(f"Session: {session_id}")
    print(f"Capturing from: {url}")
    print(f"Saving to: {out_dir}")
    print(f"Preprocess: out={args.out_size} flip={args.flip} target_px_per_mm={args.target_px_per_mm}")

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
        )


            out_jpeg = encode_jpeg(crop, quality=args.jpeg_quality)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(out_dir, f"{args.label}.{ts}.jpg")
            with open(path, "wb") as f:
                f.write(out_jpeg)

            if args.debug_every and ((i + 1) % args.debug_every == 0):
                dbg = debug_overlay(bgr, px_per_mm)
                dbg_path = os.path.join(out_dir, f"DEBUG.{args.label}.{ts}.png")
                cv2.imwrite(dbg_path, dbg)

            meta = f"px/mm={px_per_mm:.2f}" if px_per_mm else "px/mm=none"
            print(f"[{i+1}/{args.count}] saved {path} ({meta})")

        except Exception as e:
            print(f"[{i+1}/{args.count}] capture failed: {e}")

        time.sleep(args.interval)

    # Upload ONLY this session directory
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
    print("Done.")


if __name__ == "__main__":
    main()
