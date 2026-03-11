#collect_and_upload_ei.py


import argparse
import os
import time
from datetime import datetime

import requests
import edgeimpulse as ei

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def capture_one(url: str, timeout_s: int = 10) -> bytes:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "image" not in ctype and not url.endswith(".jpg"):
    
        print(f"Warning: unexpected Content-Type: {ctype}")
    return r.content

def save_image(img_bytes: bytes, out_dir: str, label: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}.{ts}.jpg"
    path = os.path.join(out_dir, filename)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path

def upload_directory_to_ei(directory: str, category: str) -> None:
    resp = ei.experimental.data.upload_directory(
        directory=directory,
        category=category,     # "training" or "testing"
        label=None,            # infer from filename prefix before first '.'
        metadata={
            "source": "esp32-cam",
            "captured_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    if len(resp.fails) > 0:
        print("Some files failed to upload:")
        for f in resp.fails:
            print(f"  - {f}")
        raise SystemExit(1)

    print(f"Uploaded {len(resp.successes)} file(s) to Edge Impulse ({category}).")

def main():
    p = argparse.ArgumentParser(description="Collect ESP32-CAM images and upload to Edge Impulse.")
    p.add_argument("--esp32", required=True, help="Base URL, e.g. http://192.168.1.50")
    p.add_argument("--endpoint", default="/capture.jpg", help="JPEG endpoint (default: /capture.jpg)")
    p.add_argument("--label", required=True, help="Label, e.g. M3, M4, M5")
    p.add_argument("--count", type=int, default=100, help="Number of images to capture")
    p.add_argument("--interval", type=float, default=3.0, help="Seconds between captures")
    p.add_argument("--out", default="dataset", help="Output directory")
    p.add_argument("--category", default="training", choices=["training", "testing"], help="Edge Impulse category")
    p.add_argument("--api-key", default=os.environ.get("EI_API_KEY"), help="Edge Impulse API key (or set EI_API_KEY env var)")
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Provide --api-key or set EI_API_KEY environment variable.")

    ei.API_KEY = args.api_key

    base = args.esp32.rstrip("/")
    url = base + args.endpoint
    out_dir = os.path.join(args.out, args.label)
    ensure_dir(out_dir)

    print(f"Capturing {args.count} image(s) from: {url}")
    print(f"Saving to: {out_dir}")
    for i in range(args.count):
        try:
            img = capture_one(url)
            path = save_image(img, out_dir, args.label)
            print(f"[{i+1}/{args.count}] saved {path}")
        except Exception as e:
            print(f"[{i+1}/{args.count}] capture failed: {e}")
        time.sleep(args.interval)

    print("Uploading to Edge Impulse...")
    upload_directory_to_ei(out_dir, args.category)
    print("Done.")

if __name__ == "__main__":
    main()
