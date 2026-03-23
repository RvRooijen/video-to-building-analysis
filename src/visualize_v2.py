"""Visualisatie met semantic segmentation + diepte-gebaseerde afmetingen."""

import cv2
import numpy as np
from pathlib import Path

from semantic_segmentation import BUILDING_LABELS

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


def render_video(
    video_path: str,
    frames_dir: str,
    depth_dir: str,
    seg_dir: str,
    output_path: str,
    id2label: dict,
    scale_factor: float = 1.0,
    aggregated: dict | None = None,
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
        writer.write(overlay)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(frame_files)}]")

    writer.release()
    print(f"Video opgeslagen: {output_path}")


if __name__ == "__main__":
    import sys
    import json
    from semantic_segmentation import SemanticSegmenter
    from aggregate import aggregate_surfaces

    video_path = sys.argv[1] if len(sys.argv) > 1 else "../input/video.mp4"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "../output/analyse_video_smooth.mp4"
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.712

    segmenter = SemanticSegmenter()

    # Gebruik smoothed segmentatie en geaggregeerde waarden
    seg_dir = "output/segmentation_smooth"
    agg = aggregate_surfaces(seg_dir, "output/depth", segmenter.id2label, scale)
    print("\nGeaggregeerde waarden:")
    for label, data in agg.items():
        print(f"  {label}: {data['area_m2']}m², {data['breedte_m']}x{data['hoogte_m']}m")

    render_video(
        video_path=video_path,
        frames_dir="output/frames",
        depth_dir="output/depth",
        seg_dir=seg_dir,
        output_path=output_path,
        id2label=segmenter.id2label,
        scale_factor=scale,
        aggregated=agg,
    )
