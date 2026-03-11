import argparse
import cv2
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--marker-mm", type=float, default=40.0, help="Measured marker side length in mm (black border to black border)")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read: {args.image}")

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, args.dict))

    # Detector parameters
    try:
        params = aruco.DetectorParameters()
    except Exception:
        params = aruco.DetectorParameters_create()

    # IMPORTANT: allow inverted markers (your temp print appears inverted)
    if hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = True

    # Detect
    try:
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(img)
    except Exception:
        corners, ids, _ = aruco.detectMarkers(img, dictionary, parameters=params)

    if ids is None or len(ids) == 0:
        print("No markers detected.")
        return

    ids = ids.flatten().tolist()

    # Find requested ID if present
    if args.id in ids:
        i = ids.index(args.id)
    else:
        i = 0  # use first detected
        print(f"Requested ID {args.id} not found; using detected ID {ids[0]} instead.")

    c = corners[i].reshape(4, 2).astype(np.float32)  # order: tl,tr,br,bl usually
    # Compute side length in pixels (average of 4 edges)
    def dist(a, b): return float(np.linalg.norm(a - b))
    px = (dist(c[0], c[1]) + dist(c[1], c[2]) + dist(c[2], c[3]) + dist(c[3], c[0])) / 4.0

    px_per_mm = px / args.marker_mm
    mm_per_px = 1.0 / px_per_mm

    print(f"Detected IDs: {ids}")
    print(f"Marker side px (avg): {px:.2f}")
    print(f"px_per_mm: {px_per_mm:.6f}  (mm_per_px: {mm_per_px:.6f})")

    # Draw overlay
    out = img.copy()
    aruco.drawDetectedMarkers(out, corners, np.array(ids, dtype=np.int32).reshape(-1, 1))
    cv2.putText(out, f"px/mm={px_per_mm:.3f}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite("aruco_debug.png", out)
    print("Wrote aruco_debug.png")

    if args.show:
        cv2.imshow("aruco_debug", out)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
