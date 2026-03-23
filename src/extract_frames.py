"""Stap 1: Extract frames uit een video met voldoende overlap en minimale blur."""

import cv2
import numpy as np
from pathlib import Path


def laplacian_variance(frame: np.ndarray) -> float:
    """Meet de scherpte van een frame via Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames(
    video_path: str,
    output_dir: str,
    interval_sec: float = 0.0,
    blur_threshold: float = 0.0,
) -> list[Path]:
    """
    Extract frames uit een video.

    Args:
        video_path: Pad naar de input video.
        output_dir: Map om frames in op te slaan.
        interval_sec: Interval in seconden tussen frames.
        blur_threshold: Minimale scherpte (Laplacian variance). Lager = waziger.

    Returns:
        Lijst met paden naar opgeslagen frames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Kan video niet openen: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * interval_sec)) if interval_sec > 0 else 1

    print(f"Video: {fps:.1f} FPS, {total_frames} frames totaal")
    if interval_sec > 0:
        print(f"Extract elke {frame_interval} frames (~{interval_sec}s interval)")
    else:
        print(f"Alle {total_frames} frames extracten")
    if blur_threshold > 0:
        print(f"Blur threshold: {blur_threshold}")

    saved = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            save = True
            if blur_threshold > 0:
                sharpness = laplacian_variance(frame)
                if sharpness < blur_threshold:
                    print(f"  Frame {frame_idx} overgeslagen (blur: {sharpness:.1f})")
                    save = False
            if save:
                path = output_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(path), frame)
                saved.append(path)

        frame_idx += 1

    cap.release()
    print(f"\n{len(saved)} frames opgeslagen in {output_dir}")
    return saved


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Gebruik: python extract_frames.py <video_path> [output_dir]")
        sys.exit(1)

    video = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "output/frames"
    extract_frames(video, out)
