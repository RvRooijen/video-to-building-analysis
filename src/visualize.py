"""Genereer een output video met vlakdetectie-overlay per frame."""

import cv2
import numpy as np
from pathlib import Path

from surface_detection import detect_surfaces

# Kleuren per label (BGR)
COLORS = {
    "wand": (255, 165, 0),     # oranje
    "vloer": (0, 200, 0),      # groen
    "plafond": (200, 0, 200),  # paars
    "deur": (0, 0, 255),       # rood
    "raam": (255, 255, 0),     # cyaan
    "onbekend": (128, 128, 128),
}


def create_overlay(frame: np.ndarray, depth: np.ndarray, scale_factor: float) -> np.ndarray:
    """Teken vlakdetectie-overlay op een frame."""
    overlay = frame.copy()
    surfaces = detect_surfaces(depth, frame, scale_factor=scale_factor)

    for s in surfaces:
        if s.pixels is None:
            continue

        # Filter hele kleine detecties (noise)
        if s.area_m2 < 0.1:
            continue

        color = COLORS.get(s.label, (128, 128, 128))

        # Semi-transparante kleuroverlay op het vlak
        mask = s.pixels.astype(np.uint8) * 255
        colored = np.zeros_like(overlay)
        colored[:] = color
        overlay = np.where(
            mask[:, :, None] > 0,
            cv2.addWeighted(overlay, 0.6, colored, 0.4, 0),
            overlay,
        )

        # Label tekst met afmetingen
        ys, xs = np.where(s.pixels)
        if len(ys) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            area = round(float(s.area_m2), 1)
            breedte = round(float(s.dimensions.get('breedte_m', 0)), 1)
            hoogte = round(float(s.dimensions.get('hoogte_m', 0)), 1)
            label = f"{s.label} {area}m2"
            dim = f"{breedte}x{hoogte}m"
            orient = s.orientation

            # Achtergrond voor tekst
            for i, text in enumerate([label, dim, orient]):
                y_pos = cy - 20 + (i * 25)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (cx - 2, y_pos - th - 4), (cx + tw + 2, y_pos + 4), (0, 0, 0), -1)
                cv2.putText(overlay, text, (cx, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Legenda rechtsboven
    y_start = 30
    for label, color in COLORS.items():
        if label == "onbekend":
            continue
        cv2.rectangle(overlay, (overlay.shape[1] - 150, y_start - 15), (overlay.shape[1] - 130, y_start), color, -1)
        cv2.putText(overlay, label, (overlay.shape[1] - 125, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_start += 25

    return overlay


def render_video(
    video_path: str,
    frames_dir: str,
    depth_dir: str,
    output_path: str,
    scale_factor: float = 1.0,
):
    """Render output video met overlays."""
    frames_dir = Path(frames_dir)
    depth_dir = Path(depth_dir)

    frame_files = sorted(frames_dir.glob("*.jpg"))
    depth_files = sorted(depth_dir.glob("*.npy"))

    if not frame_files:
        print("Geen frames gevonden!")
        return

    # Lees originele video voor FPS en grootte
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Lees eerste frame voor grootte
    first = cv2.imread(str(frame_files[0]))
    h, w = first.shape[:2]

    # Maak output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Gebruik lagere FPS zodat je elk frame goed kunt zien
    out_fps = min(fps, 4)
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (w, h))

    print(f"Rendering {len(frame_files)} frames naar {output_path} ({out_fps} FPS)...")

    for i, (frame_path, depth_path) in enumerate(zip(frame_files, depth_files)):
        frame = cv2.imread(str(frame_path))
        depth = np.load(str(depth_path))

        overlay = create_overlay(frame, depth, scale_factor)
        writer.write(overlay)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(frame_files)}]")

    writer.release()
    print(f"Video opgeslagen: {output_path}")


if __name__ == "__main__":
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else "../input/video.mp4"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "../output/analyse_video.mp4"
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.905

    render_video(
        video_path=video_path,
        frames_dir="output/frames",
        depth_dir="output/depth",
        output_path=output_path,
        scale_factor=scale,
    )
