"""Aggregeer segmentatie + diepte over alle frames voor stabiele afmetingen."""

import cv2
import numpy as np
from pathlib import Path
from scipy.stats import mode
from semantic_segmentation import BUILDING_LABELS


def aggregate_surfaces(
    seg_dir: str,
    depth_dir: str,
    id2label: dict,
    scale_factor: float,
    fx: float | None = None,
    fy: float | None = None,
) -> dict:
    """
    Bereken stabiele afmetingen per element over alle frames.

    Returns:
        dict met per label: mediaan area, breedte, hoogte, en per-frame data
    """
    seg_dir = Path(seg_dir)
    depth_dir = Path(depth_dir)

    seg_files = sorted(seg_dir.glob("*.npy"))
    depth_files = sorted(depth_dir.glob("*.npy"))

    # Verzamel metingen per label over alle frames
    measurements: dict[str, list[dict]] = {}

    for seg_path, depth_path in zip(seg_files, depth_files):
        seg_map = np.load(str(seg_path))
        depth_map = np.load(str(depth_path)) * scale_factor
        h, w = seg_map.shape

        if fx is None:
            fx = w * 0.8
        if fy is None:
            fy = h * 0.8

        for uid in np.unique(seg_map):
            uid = int(uid)
            en_label = id2label.get(uid, "unknown").lower()
            nl_label = BUILDING_LABELS.get(en_label, None)
            if nl_label is None:
                continue

            mask = seg_map == uid
            pixel_count = mask.sum()
            if pixel_count < 500:
                continue

            region_depths = depth_map[mask]
            median_depth = np.median(region_depths)
            pixel_area = (median_depth / fx) * (median_depth / fy)
            area_m2 = pixel_count * pixel_area

            ys, xs = np.where(mask)
            bbox_w = (xs.max() - xs.min()) * median_depth / fx
            bbox_h = (ys.max() - ys.min()) * median_depth / fy

            if nl_label not in measurements:
                measurements[nl_label] = []

            measurements[nl_label].append({
                "area_m2": float(area_m2),
                "breedte_m": float(bbox_w),
                "hoogte_m": float(bbox_h),
                "pixel_count": int(pixel_count),
                "frame": seg_path.name,
            })

    # Bereken stabiele waarden per label
    summary = {}
    for label, data in measurements.items():
        areas = [d["area_m2"] for d in data]
        breedtes = [d["breedte_m"] for d in data]
        hoogtes = [d["hoogte_m"] for d in data]

        summary[label] = {
            "area_m2": round(float(np.median(areas)), 1),
            "breedte_m": round(float(np.median(breedtes)), 1),
            "hoogte_m": round(float(np.median(hoogtes)), 1),
            "area_std": round(float(np.std(areas)), 2),
            "frames_seen": len(data),
            "total_frames": len(seg_files),
            "confidence": round(len(data) / len(seg_files), 2),
        }

    return summary


def smooth_segmentation(
    seg_dir: str,
    output_dir: str,
    window_size: int = 7,
) -> list[Path]:
    """
    Temporeel smoothen van segmentatie-maps.

    Per pixel: neem de meest voorkomende klasse in een venster van N frames.
    Dit elimineert flikkering tussen frames.
    """
    seg_dir = Path(seg_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_files = sorted(seg_dir.glob("*.npy"))
    n = len(seg_files)

    print(f"Smoothing {n} segmentatie-maps met window={window_size}...")

    # Laad alle maps in memory
    all_maps = np.stack([np.load(str(f)) for f in seg_files])  # (N, H, W)
    h, w = all_maps.shape[1], all_maps.shape[2]

    saved = []
    half = window_size // 2

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = all_maps[start:end]  # (window, H, W)

        # Mode per pixel over de tijdsas (meest voorkomende klasse)
        smoothed = mode(window, axis=0, keepdims=False).mode

        out_path = output_dir / seg_files[i].name
        np.save(str(out_path), smoothed)
        saved.append(out_path)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{n}]")

    print(f"{len(saved)} smoothed maps opgeslagen in {output_dir}")
    return saved


if __name__ == "__main__":
    import json
    from semantic_segmentation import SemanticSegmenter

    segmenter = SemanticSegmenter()

    print("\n=== Temporal smoothing ===")
    smooth_segmentation("output/segmentation", "output/segmentation_smooth", window_size=7)

    print("\n=== Aggregatie ===")
    summary = aggregate_surfaces(
        "output/segmentation_smooth", "output/depth",
        segmenter.id2label, scale_factor=0.712,
    )

    print("\nResultaat:")
    for label, data in summary.items():
        print(f"  {label:10s} | {data['area_m2']:5.1f} m² | "
              f"{data['breedte_m']:.1f}x{data['hoogte_m']:.1f}m | "
              f"gezien in {data['confidence']*100:.0f}% van frames | "
              f"σ={data['area_std']:.2f}")

    with open("output/aggregated_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
