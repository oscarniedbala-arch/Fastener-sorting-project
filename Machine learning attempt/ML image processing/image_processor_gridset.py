# bgr = decode from JPEG bytes
px_per_mm = estimate_px_per_mm_from_squares(bgr, square_mm=5.0)


target_px_per_mm = 10.0  # e.g.  10 px per mm

crop = crop_and_scale_normalize(
    bgr,
    out_size=128,
    target_px_per_mm=target_px_per_mm,
    px_per_mm=px_per_mm
)

# gauge estimate:
d_mm = measure_diameter_mm(bgr, px_per_mm)
label = classify_m_gauge(d_mm)
