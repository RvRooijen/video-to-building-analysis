"""Draai de volledige pipeline in één commando."""

import sys
from pathlib import Path

from extract_frames import extract_frames
from depth_estimation import DepthEstimator
from semantic_segmentation import SemanticSegmenter
from aggregate import smooth_segmentation, aggregate_surfaces
from calibration import calibrate_from_video


def run(video_path: str, output_dir: str = "../output"):
    output_dir = Path(output_dir)
    frames_dir = "output/frames"
    depth_dir = "output/depth"
    seg_dir = "output/segmentation"
    seg_smooth_dir = "output/segmentation_smooth"

    # 1. Frames
    print("=" * 60)
    print("STAP 1/6: Frames extracten")
    print("=" * 60)
    extract_frames(video_path, frames_dir)

    # 2. Diepte
    print("\n" + "=" * 60)
    print("STAP 2/6: Diepteschatting")
    print("=" * 60)
    estimator = DepthEstimator()
    estimator.process_frames(frames_dir, depth_dir)

    # 3. Segmentatie
    print("\n" + "=" * 60)
    print("STAP 3/6: Semantic segmentation")
    print("=" * 60)
    segmenter = SemanticSegmenter()
    segmenter.process_frames(frames_dir, seg_dir)

    # 4. Smoothing
    print("\n" + "=" * 60)
    print("STAP 4/6: Temporal smoothing")
    print("=" * 60)
    smooth_segmentation(seg_dir, seg_smooth_dir)

    # 5. Kalibratie
    print("\n" + "=" * 60)
    print("STAP 5/6: Schaalcalibratie")
    print("=" * 60)
    scale, cal_info = calibrate_from_video(frames_dir, depth_dir)
    if "error" in cal_info:
        print(f"  Geen ArUco marker gevonden, fallback schaalfactor 1.0")
        print(f"  Tip: print een marker via 'python calibration.py generate'")
        scale = 1.0

    # 6. Aggregatie + video
    print("\n" + "=" * 60)
    print("STAP 6/6: Aggregatie en video renderen")
    print("=" * 60)
    agg = aggregate_surfaces(seg_smooth_dir, depth_dir, segmenter.id2label, scale)

    print("\nResultaat:")
    for label, data in agg.items():
        print(f"  {label:10s} | {data['area_m2']:5.1f} m² | "
              f"{data['breedte_m']:.1f}x{data['hoogte_m']:.1f}m")

    from visualize_v2 import render_video
    video_out = str(output_dir / "result.mp4")
    render_video(
        video_path=video_path,
        frames_dir=frames_dir,
        depth_dir=depth_dir,
        seg_dir=seg_smooth_dir,
        output_path=video_out,
        id2label=segmenter.id2label,
        scale_factor=scale,
        aggregated=agg,
    )

    # Converteer naar H.264 voor compatibiliteit
    import subprocess
    h264_out = str(output_dir / "result_h264.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_out,
        "-c:v", "libx264", "-profile:v", "baseline",
        "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
        h264_out,
    ], capture_output=True)

    print(f"\nKlaar!")
    print(f"  Video: {video_out}")
    print(f"  Video (H.264): {h264_out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Gebruik: python run.py <pad_naar_video>")
        print("Voorbeeld: python run.py ../input/video.mp4")
        sys.exit(1)

    run(sys.argv[1])
