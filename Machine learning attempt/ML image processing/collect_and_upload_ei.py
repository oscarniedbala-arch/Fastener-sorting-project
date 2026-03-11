import argparse
import os
import time
from datetime import datetime

import requests
import edgeimpulse as ei


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def capture_jpeg(url: str, timeout_s: int = 10) -> bytes:
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.content


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--esp32", required=True, help="e.g. http://192.168.1.50")
    p.add_argument("--endpoint", default="/capture.jpg")
    p.add_argument("--label", required=True, help="e.g. M3, M4, M5")
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--interval", type=float, default=3.0)
    p.add_argument("--category", default="training", choices=["training", "testing"])
    p.add_argument("--api-key", default=os.environ.get("EI_API_KEY"))
    p.add_argument("--sessions-root", default="sessions")
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Provide --api-key or set EI_API_KEY.")

    # Use a unique per-run session folder
    session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = os.path.join(args.sessions_root, session_id, args.label)
    ensure_dir(out_dir)

    base = args.esp32.rstrip("/")
    url = base + args.endpoint

    print(f"Session: {session_id}")
    print(f"Capturing from: {url}")
    print(f"Saving to: {out_dir}")

    for i in range(args.count):
        try:
            img = capture_jpeg(url)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(out_dir, f"{args.label}.{ts}.jpg")
            with open(path, "wb") as f:
                f.write(img)
            print(f"[{i+1}/{args.count}] saved {path}")
        except Exception as e:
            print(f"[{i+1}/{args.count}] capture failed: {e}")
        time.sleep(args.interval)

    # Upload ONLY this session directory
    ei.API_KEY = args.api_key
    print("Uploading session folder to Edge Impulse...")

    resp = ei.experimental.data.upload_directory(
        directory=out_dir,
        category=args.category,
        label=args.label,  # explicit label to avoid any inference issues
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
