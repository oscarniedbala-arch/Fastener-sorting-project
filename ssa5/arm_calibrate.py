import argparse
import sys
import time
from typing import Optional

import requests

# Windows keypress
try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


class ESP32Arm:
    def __init__(self, ip: str, timeout: float = 2.0):
        self.base = f"http://{ip}"
        self.s = requests.Session()
        self.timeout = timeout

    def status(self) -> dict:
        r = self.s.get(f"{self.base}/api/status", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def set_servo(self, ch: int, us: int, speed: int = 80) -> None:
        # speed is handled by your ESP32 firmware (slew rate). If your firmware ignores it, it’s harmless.
        params = {"ch": ch, "val": us, "speed": speed}
        r = self.s.get(f"{self.base}/api/servo", params=params, timeout=self.timeout)
        r.raise_for_status()

    def set_all_neutral(self, neutral_us: int = 1500, speed: int = 60, channels: int = 4) -> None:
        # Uses /api/servos if available; otherwise falls back to per-channel.
        # If your firmware doesn’t implement /api/servos, this still works (fallback).
        params = {"speed": speed}
        for i in range(channels):
            params[f"c{i}"] = neutral_us
        try:
            r = self.s.get(f"{self.base}/api/servos", params=params, timeout=self.timeout)
            if r.status_code == 200:
                return
        except Exception:
            pass

        # fallback
        for i in range(channels):
            self.set_servo(i, neutral_us, speed=speed)
            time.sleep(0.05)


def print_help():
    print(
        "\nControls (Nudge Mode):\n"
        "  [ or A  : decrease pulse\n"
        "  ] or D  : increase pulse\n"
        "  -       : decrease step size\n"
        "  +       : increase step size\n"
        "  s       : run sweep min->max->min for current channel\n"
        "  n       : next channel\n"
        "  p       : previous channel\n"
        "  g       : go to neutral (1500us) for current channel\n"
        "  h       : help\n"
        "  q       : quit (optionally neutral all)\n"
    )


def read_key() -> Optional[str]:
    if not HAS_MSVCRT:
        return None
    if not msvcrt.kbhit():
        return None
    k = msvcrt.getch()
    try:
        return k.decode("utf-8", errors="ignore")
    except Exception:
        return None


def sweep(arm: ESP32Arm, ch: int, lo: int, hi: int, step: int, dwell: float, speed: int):
    print(f"\nSWEEP CH{ch}: {lo} -> {hi} -> {lo}  step={step}us  dwell={dwell:.2f}s  speed={speed}")
    # up
    for us in range(lo, hi + 1, step):
        arm.set_servo(ch, us, speed=speed)
        print(f"CH{ch} = {us} us")
        time.sleep(dwell)
    # down
    for us in range(hi, lo - 1, -step):
        arm.set_servo(ch, us, speed=speed)
        print(f"CH{ch} = {us} us")
        time.sleep(dwell)
    print("SWEEP DONE\n")


def main():
    ap = argparse.ArgumentParser(description="ESP32-PCA9685 Arm Calibration Tool")
    ap.add_argument("--esp32", default="192.168.0.202", help="ESP32 IP address (default: 192.168.0.202)")
    ap.add_argument("--channels", type=int, default=4, help="Number of servo channels to calibrate (default: 4)")
    ap.add_argument("--start-ch", type=int, default=0, help="Start channel index (default: 0)")
    ap.add_argument("--min", dest="min_us", type=int, default=700, help="Min pulse (us) (default: 700)")
    ap.add_argument("--max", dest="max_us", type=int, default=2300, help="Max pulse (us) (default: 2300)")
    ap.add_argument("--neutral", type=int, default=1500, help="Neutral pulse (us) (default: 1500)")
    ap.add_argument("--step", type=int, default=10, help="Nudge step in us (default: 10)")
    ap.add_argument("--speed", type=int, default=80, help="Servo slew speed (default: 80)")
    ap.add_argument("--dwell", type=float, default=0.08, help="Sweep dwell seconds (default: 0.08)")
    ap.add_argument("--neutral-all-on-exit", action="store_true", help="Send all channels to neutral on exit")
    args = ap.parse_args()

    if not HAS_MSVCRT:
        print("This script uses msvcrt for key control and is intended for Windows PowerShell/CMD.")

    arm = ESP32Arm(args.esp32)

    print("\n============================================================")
    print(" ARM CALIBRATION TOOL (ESP32-CAM + PCA9685)")
    print("============================================================\n")

    # Connectivity check
    try:
        st = arm.status()
        print(f"ESP32 OK | IP reported: {st.get('ip')} | RSSI: {st.get('rssi')} | heap: {st.get('free_heap')}")
    except Exception as e:
        print(f"ERROR: Could not reach ESP32 /api/status at {args.esp32}: {e}")
        sys.exit(1)

    print("\nSAFETY:")
    print(" - Keep the arm clear of its base before commanding movement.")
    print(" - Start with small nudges, find safe min/max for each joint.")
    print(" - If anything binds/clicks: stop and reduce range immediately.\n")

    print_help()

    ch = clamp(args.start_ch, 0, max(0, args.channels - 1))
    lo = args.min_us
    hi = args.max_us
    neutral = args.neutral
    step = max(1, args.step)
    speed = max(1, args.speed)
    dwell = max(0.01, args.dwell)

    # We do NOT move anything automatically on start.
    current_us = neutral

    print(f"\nSelected CH{ch}. Current target = {current_us}us (not sent yet).")
    print("Press 'g' to go neutral, or nudge with [ ] / A D.\n")

    try:
        while True:
            k = read_key()
            if k is None:
                time.sleep(0.01)
                continue

            k = k.lower()

            if k in ("q",):
                print("\nQuit requested.")
                break

            if k in ("h",):
                print_help()
                continue

            if k in ("n",):
                ch = (ch + 1) % args.channels
                current_us = neutral
                print(f"\n--- Now CH{ch}. Target reset to {current_us}us (not sent). ---")
                continue

            if k in ("p",):
                ch = (ch - 1) % args.channels
                current_us = neutral
                print(f"\n--- Now CH{ch}. Target reset to {current_us}us (not sent). ---")
                continue

            if k in ("g",):
                current_us = clamp(neutral, lo, hi)
                arm.set_servo(ch, current_us, speed=speed)
                print(f"CH{ch} = {current_us} us (neutral)")
                continue

            if k in ("[", "a"):
                current_us = clamp(current_us - step, lo, hi)
                arm.set_servo(ch, current_us, speed=speed)
                print(f"CH{ch} = {current_us} us  (step={step}, speed={speed})")
                continue

            if k in ("]", "d"):
                current_us = clamp(current_us + step, lo, hi)
                arm.set_servo(ch, current_us, speed=speed)
                print(f"CH{ch} = {current_us} us  (step={step}, speed={speed})")
                continue

            if k == "-":
                step = max(1, step - 1)
                print(f"Step = {step} us")
                continue

            if k in ("+", "="):
                step = min(200, step + 1)
                print(f"Step = {step} us")
                continue

            if k == "s":
                sweep(arm, ch, lo, hi, step=max(step, 5), dwell=dwell, speed=speed)
                continue

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt - exiting.")

    finally:
        if args.neutral_all_on_exit:
            print("\nSending all channels to neutral...")
            try:
                arm.set_all_neutral(neutral_us=neutral, speed=60, channels=args.channels)
                print("Neutral sent.")
            except Exception as e:
                print(f"Neutral-all failed: {e}")

        print("Done.")


if __name__ == "__main__":
    main()
