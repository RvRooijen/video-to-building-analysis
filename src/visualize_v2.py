"""Visualisatie met semantic segmentation + diepte-gebaseerde afmetingen."""

import cv2
import numpy as np
from pathlib import Path

from semantic_segmentation import BUILDING_LABELS
from calibration import detect_marker_in_frame

# Kleuren per NL label (BGR)
COLORS = {
    "wand": (255, 165, 0),
    "vloer": (0, 200, 0),
    "plafond": (200, 0, 200),
    "deur": (0, 0, 255),
    "raam": (255, 255, 0),
    "kolom": (0, 255, 255),
    "trap": (128, 0, 255),
    "balk": (255, 128, 0),
    "leiding": (0, 128, 255),
}

DEFAULT_COLOR = (128, 128, 128)


def create_overlay(
    frame: np.ndarray,
    seg_map: np.ndarray,
    depth_map: np.ndarray,
    id2label: dict,
    scale_factor: float,
    fx: float | None = None,
    fy: float | None = None,
    aggregated: dict | None = None,
) -> np.ndarray:
    """Teken semantic segmentation overlay met afmetingen (geaggregeerd of per-frame)."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    if fx is None:
        fx = w * 0.8
    if fy is None:
        fy = h * 0.8

    depth_scaled = depth_map * scale_factor

    unique_ids = np.unique(seg_map)

    for uid in unique_ids:
        uid = int(uid)
        en_label = id2label.get(uid, "unknown").lower()
        nl_label = BUILDING_LABELS.get(en_label, None)

        if nl_label is None:
            continue

        mask = seg_map == uid
        pixel_count = mask.sum()

        if pixel_count < 500:
            continue

        color = COLORS.get(nl_label, DEFAULT_COLOR)

        # Semi-transparante overlay
        colored = np.zeros_like(overlay)
        colored[:] = color
        overlay = np.where(
            mask[:, :, None],
            cv2.addWeighted(overlay, 0.55, colored, 0.45, 0),
            overlay,
        )

        # Gebruik geaggregeerde waarden als beschikbaar, anders per-frame
        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())

        if aggregated and nl_label in aggregated:
            agg = aggregated[nl_label]
            area_m2 = agg["area_m2"]
            bbox_w = agg["breedte_m"]
            bbox_h = agg["hoogte_m"]
        else:
            region_depths = depth_scaled[mask]
            if len(region_depths) == 0:
                continue
            mean_depth = np.median(region_depths)
            pixel_area = (mean_depth / fx) * (mean_depth / fy)
            area_m2 = pixel_count * pixel_area
            bbox_w = (xs.max() - xs.min()) * mean_depth / fx
            bbox_h = (ys.max() - ys.min()) * mean_depth / fy

        lines = [
            f"{nl_label} {area_m2:.1f}m2",
            f"{bbox_w:.1f}x{bbox_h:.1f}m",
        ]

        for i, text in enumerate(lines):
            y_pos = cy - 10 + (i * 22)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (cx - 2, y_pos - th - 4), (cx + tw + 2, y_pos + 4), (0, 0, 0), -1)
            cv2.putText(overlay, text, (cx, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Legenda
    y_start = 30
    for nl_label, color in COLORS.items():
        cv2.rectangle(overlay, (w - 150, y_start - 15), (w - 130, y_start), color, -1)
        cv2.putText(overlay, nl_label, (w - 125, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_start += 25

    return overlay


def draw_aruco_overlay(
    overlay: np.ndarray,
    frame: np.ndarray,
    scale_factor: float,
    cal_info: dict | None = None,
) -> np.ndarray:
    """Teken ArUco marker detectie en schaalinformatie op het frame."""
    corners, pixel_size = detect_marker_in_frame(frame)

    if corners is not None:
        # Teken marker contour (groen)
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

        # Teken hoekpunten
        for pt in corners.astype(np.int32):
            cv2.circle(overlay, tuple(pt), 5, (0, 255, 0), -1)

        # Label bij de marker
        cx, cy = int(corners[:, 0].mean()), int(corners[:, 1].mean())
        cv2.putText(overlay, "ArUco", (cx - 20, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Schaalinformatie linksboven
    h, w = overlay.shape[:2]
    y = h - 20
    scale_text = f"Schaal: x{scale_factor:.3f}"
    marker_text = "Marker: GEVONDEN" if corners is not None else "Marker: niet in beeld"
    marker_color = (0, 255, 0) if corners is not None else (0, 0, 200)

    for text, color in [(scale_text, (255, 255, 255)), (marker_text, marker_color)]:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (8, y - th - 4), (12 + tw + 2, y + 4), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y -= 25

    return overlay


def render_video(
    video_path: str,
    frames_dir: str,
    depth_dir: str,
    seg_dir: str,
    output_path: str,
    id2label: dict,
    scale_factor: float = 1.0,
    aggregated: dict | None = None,
    cal_info: dict | None = None,
):
    """Render output video met semantic segmentation overlay."""
    frames_dir = Path(frames_dir)
    depth_dir = Path(depth_dir)
    seg_dir = Path(seg_dir)

    frame_files = sorted(frames_dir.glob("*.jpg"))
    depth_files = sorted(depth_dir.glob("*.npy"))
    seg_files = sorted(seg_dir.glob("*.npy"))

    if not frame_files or not seg_files:
        print("Geen frames of segmentatie gevonden!")
        return

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    first = cv2.imread(str(frame_files[0]))
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    print(f"Rendering {len(frame_files)} frames naar {output_path}...")

    for i, (frame_path, depth_path, seg_path) in enumerate(zip(frame_files, depth_files, seg_files)):
        frame = cv2.imread(str(frame_path))
        depth = np.load(str(depth_path))
        seg_map = np.load(str(seg_path))

        overlay = create_overlay(frame, seg_map, depth, id2label, scale_factor, aggregated=aggregated)
        overlay = draw_aruco_overlay(overlay, frame, scale_factor, cal_info)
        writer.write(overlay)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(frame_files)}]")

    writer.release()
    print(f"Video opgeslagen: {output_path}")


def depth_to_colormap(depth_map: np.ndarray, scale_factor: float) -> np.ndarray:
    """Converteer dieptemap naar een kleurenvisualisatie met afstandslabels."""
    depth_scaled = depth_map * scale_factor

    # Normaliseer naar 0-255 (dichtbij=warm, ver=koud)
    valid = depth_scaled[depth_scaled > 0]
    if len(valid) == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)

    d_min, d_max = np.percentile(valid, [2, 98])
    normalized = np.clip((depth_scaled - d_min) / (d_max - d_min + 1e-6), 0, 1)
    # Inverteer: dichtbij = rood/geel, ver = blauw
    normalized = 1.0 - normalized
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    # Afstandslabels op vaste punten
    h, w = depth_map.shape
    for row_frac in [0.25, 0.5, 0.75]:
        for col_frac in [0.25, 0.5, 0.75]:
            y, x = int(h * row_frac), int(w * col_frac)
            d = depth_scaled[y, x]
            if d > 0:
                text = f"{d:.1f}m"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(colored, (x - 2, y - th - 4), (x + tw + 2, y + 4), (0, 0, 0), -1)
                cv2.putText(colored, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Schaalbalk rechtsonder
    bar_w, bar_h = 20, h - 60
    bar_x = w - 40
    bar_y = 30
    for i in range(bar_h):
        frac = i / bar_h
        color_val = int((1.0 - frac) * 255)
        row_color = cv2.applyColorMap(np.array([[color_val]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0][0]
        cv2.line(colored, (bar_x, bar_y + i), (bar_x + bar_w, bar_y + i), row_color.tolist(), 1)

    cv2.putText(colored, f"{d_min:.1f}m", (bar_x - 45, bar_y + bar_h + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(colored, f"{d_max:.1f}m", (bar_x - 45, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Titel
    cv2.putText(colored, "Depth Anything V2", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return colored


def interactive_viewer(
    frames_dir: str,
    depth_dir: str,
    seg_dir: str,
    id2label: dict,
    scale_factor: float = 1.0,
    aggregated: dict | None = None,
):
    """
    Interactieve viewer om te schakelen tussen weergaven.

    Toetsen:
        1 = Segmentatie overlay (standaard)
        2 = Depth map
        3 = Depth + segmentatie naast elkaar
        ←/→ = Vorig/volgend frame
        q/ESC = Sluiten
    """
    frames_dir = Path(frames_dir)
    depth_dir = Path(depth_dir)
    seg_dir = Path(seg_dir)

    frame_files = sorted(frames_dir.glob("*.jpg"))
    depth_files = sorted(depth_dir.glob("*.npy"))
    seg_files = sorted(seg_dir.glob("*.npy"))

    n = min(len(frame_files), len(depth_files), len(seg_files))
    if n == 0:
        print("Geen data gevonden!")
        return

    idx = 0
    view_mode = 1  # 1=segmentatie, 2=depth, 3=naast elkaar
    mode_names = {1: "Segmentatie", 2: "Depth map", 3: "Naast elkaar"}

    print(f"Viewer gestart met {n} frames")
    print("  1 = Segmentatie overlay")
    print("  2 = Depth map")
    print("  3 = Naast elkaar")
    print("  ←/→ = Navigeren, q = Sluiten")

    cv2.namedWindow("Viewer", cv2.WINDOW_NORMAL)

    while True:
        frame = cv2.imread(str(frame_files[idx]))
        depth = np.load(str(depth_files[idx]))
        seg_map = np.load(str(seg_files[idx]))

        seg_overlay = create_overlay(frame, seg_map, depth, id2label, scale_factor, aggregated=aggregated)
        seg_overlay = draw_aruco_overlay(seg_overlay, frame, scale_factor)

        depth_vis = depth_to_colormap(depth, scale_factor)
        depth_vis = draw_aruco_overlay(depth_vis, frame, scale_factor)

        if view_mode == 1:
            display = seg_overlay
        elif view_mode == 2:
            display = depth_vis
        else:
            # Naast elkaar: resize beide tot halve breedte
            h, w = frame.shape[:2]
            half_w = w // 2
            left = cv2.resize(seg_overlay, (half_w, h))
            right = cv2.resize(depth_vis, (half_w, h))
            display = np.hstack([left, right])

        # Frame info bovenbalk
        info = f"[{idx+1}/{n}] {frame_files[idx].name} | Modus: {mode_names[view_mode]} | 1/2/3=wissel  <-/-> nav  q=quit"
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(display, (0, 0), (tw + 10, th + 12), (0, 0, 0), -1)
        cv2.putText(display, info, (5, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Viewer", display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:  # q of ESC
            break
        elif key == ord('1'):
            view_mode = 1
        elif key == ord('2'):
            view_mode = 2
        elif key == ord('3'):
            view_mode = 3
        elif key == 83 or key == ord('d'):  # → of d
            idx = min(idx + 1, n - 1)
        elif key == 81 or key == ord('a'):  # ← of a
            idx = max(idx - 1, 0)
        elif key == ord('.'):  # 10 frames vooruit
            idx = min(idx + 10, n - 1)
        elif key == ord(','):  # 10 frames terug
            idx = max(idx - 10, 0)

    cv2.destroyAllWindows()
    print("Viewer gesloten.")


if __name__ == "__main__":
    import sys
    from semantic_segmentation import SemanticSegmenter
    from aggregate import aggregate_surfaces

    segmenter = SemanticSegmenter()
    seg_dir = "output/segmentation_smooth"
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 2.3203

    agg = aggregate_surfaces(seg_dir, "output/depth", segmenter.id2label, scale)
    print("\nGeaggregeerde waarden:")
    for label, data in agg.items():
        print(f"  {label}: {data['area_m2']}m², {data['breedte_m']}x{data['hoogte_m']}m")

    interactive_viewer(
        frames_dir="output/frames",
        depth_dir="output/depth",
        seg_dir=seg_dir,
        id2label=segmenter.id2label,
        scale_factor=scale,
        aggregated=agg,
    )
