"""Automatische schaalcalibratie via ArUco marker op A4 papier."""

import cv2
import numpy as np
from pathlib import Path


# Bekende maat van het A4 papier waarop de marker geprint is
A4_WIDTH_M = 0.210
A4_HEIGHT_M = 0.297

# ArUco dictionary — we gebruiken 4x4_50 (simpel, makkelijk te printen)
ARUCO_DICT = cv2.aruco.DICT_4X4_50


def generate_marker(output_path: str = "marker.png", marker_id: int = 0, size_px: int = 500):
    """Genereer een printbare ArUco marker."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)

    # Voeg witte rand toe (nodig voor detectie)
    border = 80
    with_border = np.ones((size_px + 2 * border, size_px + 2 * border), dtype=np.uint8) * 255
    with_border[border:border + size_px, border:border + size_px] = marker

    cv2.imwrite(output_path, with_border)
    print(f"Marker opgeslagen: {output_path}")
    print(f"Print dit op A4 papier en plak het op een muur in de ruimte.")
    return output_path


def detect_marker_in_frame(frame: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    """
    Detecteer ArUco marker in een frame.

    Returns:
        corners: 4 hoekpunten van de marker, of None
        pixel_size: gemiddelde zijlengte in pixels, of None
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is None or len(ids) == 0:
        return None, None

    # Pak de eerste gedetecteerde marker
    marker_corners = corners[0][0]  # (4, 2) array

    # Bereken zijlengtes in pixels
    side_lengths = []
    for i in range(4):
        p1 = marker_corners[i]
        p2 = marker_corners[(i + 1) % 4]
        side_lengths.append(np.linalg.norm(p2 - p1))

    avg_side_px = np.mean(side_lengths)

    return marker_corners, avg_side_px


def calibrate_from_marker(
    frame: np.ndarray,
    depth_map: np.ndarray,
    marker_real_size_m: float,
    fx: float | None = None,
    fy: float | None = None,
) -> tuple[float, dict]:
    """
    Bereken schaalfactor uit een gedetecteerde ArUco marker.

    Args:
        frame: RGB frame
        depth_map: bijbehorende dieptemap
        marker_real_size_m: werkelijke grootte van de marker in meters
        fx, fy: focal lengths (geschat als None)

    Returns:
        scale_factor, info dict
    """
    h, w = frame.shape[:2]
    if fx is None:
        fx = w * 0.8
    if fy is None:
        fy = h * 0.8

    corners, pixel_size = detect_marker_in_frame(frame)

    if corners is None:
        return 1.0, {"error": "Geen marker gedetecteerd"}

    # Diepte op de marker-locatie
    center = corners.mean(axis=0).astype(int)
    cx, cy = center[0], center[1]

    # Sample diepte in een klein gebied rond het center
    margin = 10
    y_start = max(0, cy - margin)
    y_end = min(h, cy + margin)
    x_start = max(0, cx - margin)
    x_end = min(w, cx + margin)

    marker_depth = np.median(depth_map[y_start:y_end, x_start:x_end])

    # Geschatte grootte van de marker in meters (via diepte)
    estimated_size_m = pixel_size * marker_depth / fx

    # Schaalfactor
    scale_factor = marker_real_size_m / estimated_size_m if estimated_size_m > 0 else 1.0

    info = {
        "marker_pixel_size": round(float(pixel_size), 1),
        "marker_depth_m": round(float(marker_depth), 3),
        "estimated_size_m": round(float(estimated_size_m), 4),
        "real_size_m": marker_real_size_m,
        "scale_factor": round(float(scale_factor), 4),
        "marker_center": [int(cx), int(cy)],
    }

    return scale_factor, info


def calibrate_from_video(
    frames_dir: str,
    depth_dir: str,
    marker_real_size_m: float = 0.18,  # standaard ArUco marker op A4 (~18cm)
) -> tuple[float, dict]:
    """
    Zoek de beste schaalfactor over alle frames.

    Returns:
        Mediaan schaalfactor en info van het beste frame.
    """
    frames_dir = Path(frames_dir)
    depth_dir = Path(depth_dir)

    frame_files = sorted(frames_dir.glob("*.jpg"))
    depth_files = sorted(depth_dir.glob("*.npy"))

    scales = []
    best_info = None
    best_size = 0

    print(f"Zoeken naar ArUco marker in {len(frame_files)} frames...")

    for frame_path, depth_path in zip(frame_files, depth_files):
        frame = cv2.imread(str(frame_path))
        depth = np.load(str(depth_path))

        scale, info = calibrate_from_marker(frame, depth, marker_real_size_m)

        if "error" in info:
            continue

        scales.append(scale)

        if info["marker_pixel_size"] > best_size:
            best_size = info["marker_pixel_size"]
            best_info = info
            best_info["frame"] = frame_path.name

    if not scales:
        print("  Geen marker gevonden in enig frame!")
        return 1.0, {"error": "Marker niet gevonden"}

    median_scale = float(np.median(scales))

    print(f"  Marker gevonden in {len(scales)}/{len(frame_files)} frames")
    print(f"  Mediaan schaalfactor: {median_scale:.4f}")
    print(f"  Beste frame: {best_info['frame']}")
    print(f"  Marker grootte: {best_info['estimated_size_m']*1000:.0f}mm geschat, "
          f"{marker_real_size_m*1000:.0f}mm werkelijk")

    return median_scale, best_info


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        output = sys.argv[2] if len(sys.argv) > 2 else "../output/aruco_marker.png"
        generate_marker(output)
    else:
        scale, info = calibrate_from_video("output/frames", "output/depth")
        print(f"\nSchaalfactor: {scale}")
