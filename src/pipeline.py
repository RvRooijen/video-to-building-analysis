"""Hoofdpipeline: Video → Analyse-data."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from extract_frames import extract_frames
from depth_estimation import DepthEstimator
from surface_detection import detect_surfaces, Surface


def _deduplicate_surfaces(
    surfaces: list[dict],
    center_threshold: float = 0.5,
    normal_threshold: float = 0.3,
) -> list[dict]:
    """
    Voeg vlakken samen die in meerdere frames voorkomen.

    Twee vlakken worden als 'zelfde' beschouwd als:
    - Ze hetzelfde label hebben
    - Hun centers dicht bij elkaar liggen (< center_threshold meter)
    - Hun normalen vergelijkbaar zijn (< normal_threshold verschil)

    Bij samenvoegen wordt het vlak met de grootste oppervlakte behouden.
    """
    if not surfaces:
        return []

    merged = []
    used = [False] * len(surfaces)

    for i, s1 in enumerate(surfaces):
        if used[i]:
            continue

        group = [s1]
        used[i] = True

        for j, s2 in enumerate(surfaces):
            if used[j] or j <= i:
                continue

            if s1["label"] != s2["label"]:
                continue

            c1 = np.array(s1["center"])
            c2 = np.array(s2["center"])
            n1 = np.array(s1["normal"])
            n2 = np.array(s2["normal"])

            center_dist = np.linalg.norm(c1 - c2)
            normal_diff = np.linalg.norm(n1 - n2)

            if center_dist < center_threshold and normal_diff < normal_threshold:
                group.append(s2)
                used[j] = True

        # Behoud het vlak met de grootste oppervlakte uit de groep
        best = max(group, key=lambda s: s["area_m2"])
        best["seen_in_frames"] = len(group)
        merged.append(best)

    return merged


def run_pipeline(
    video_path: str,
    reference_height_m: float = 2.10,
    output_dir: str = "output",
) -> dict:
    """
    Volledige pipeline: video → gebouwdata.

    Args:
        video_path: Pad naar de input video.
        reference_height_m: Bekende maat voor schaalcorrectie (bijv. deurhoogte).
        output_dir: Output directory.

    Returns:
        Dictionary met alle gedetecteerde elementen en hun eigenschappen.
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    depth_dir = output_dir / "depth"

    # === Stap 1: Frames extracten ===
    print("=" * 60)
    print("STAP 1: Frames extracten uit video")
    print("=" * 60)
    frame_paths = extract_frames(video_path, str(frames_dir))

    if not frame_paths:
        print("Geen bruikbare frames gevonden!")
        return {}

    # === Stap 2: Diepteschatting ===
    print("\n" + "=" * 60)
    print("STAP 2: Diepte schatten per frame")
    print("=" * 60)
    estimator = DepthEstimator()
    depth_paths = estimator.process_frames(str(frames_dir), str(depth_dir))

    # === Stap 3: Vlakken detecteren ===
    print("\n" + "=" * 60)
    print("STAP 3: Vlakken detecteren en classificeren")
    print("=" * 60)

    all_surfaces: list[dict] = []

    # Schaalcorrectie: eerste frame gebruiken om verhouding te bepalen
    # We nemen aan dat de grootste verticale extent in het eerste frame
    # overeenkomt met de referentiehoogte (deurhoogte).
    # TODO: interactieve selectie van het referentie-element
    first_depth = np.load(str(depth_paths[0]))
    estimated_height = first_depth.max() - first_depth.min()
    scale_factor = reference_height_m / estimated_height if estimated_height > 0 else 1.0
    print(f"  Schaalcorrectie: {scale_factor:.3f} "
          f"(geschat {estimated_height:.2f}m → referentie {reference_height_m}m)")

    for frame_path, depth_path in zip(frame_paths, depth_paths):
        depth = np.load(str(depth_path))
        frame = cv2.imread(str(frame_path))

        surfaces = detect_surfaces(
            depth_map=depth,
            rgb_frame=frame,
            scale_factor=scale_factor,
        )

        for s in surfaces:
            all_surfaces.append({
                "frame": frame_path.name,
                "label": s.label,
                "area_m2": s.area_m2,
                "orientation": s.orientation,
                "breedte_m": s.dimensions.get("breedte_m", 0),
                "hoogte_m": s.dimensions.get("hoogte_m", 0),
                "center": s.center.tolist(),
                "normal": s.normal.tolist(),
            })

    # === Stap 4: Deduplicatie — vlakken over frames samenvoegen ===
    print("\n" + "=" * 60)
    print("STAP 4: Vlakken dedupliceren over frames")
    print("=" * 60)

    unique_surfaces = _deduplicate_surfaces(all_surfaces)
    print(f"  {len(all_surfaces)} ruwe vlakken → {len(unique_surfaces)} unieke vlakken")

    # === Stap 5: Resultaten aggregeren ===
    print("\n" + "=" * 60)
    print("STAP 5: Resultaten samenvatten")
    print("=" * 60)

    summary = {
        "input_video": str(video_path),
        "reference_height_m": reference_height_m,
        "scale_factor": round(scale_factor, 3),
        "frames_processed": len(frame_paths),
        "raw_surfaces": len(all_surfaces),
        "unique_surfaces": len(unique_surfaces),
        "surfaces": unique_surfaces,
        "totals": {},
    }

    # Aggregeer per type
    for label in ["wand", "vloer", "plafond"]:
        matching = [s for s in unique_surfaces if s["label"] == label]
        if matching:
            total_area = sum(s["area_m2"] for s in matching)
            summary["totals"][label] = {
                "count": len(matching),
                "total_area_m2": round(total_area, 2),
            }

    # Oriëntatie-verdeling voor wanden
    wanden = [s for s in unique_surfaces if s["label"] == "wand"]
    if wanden:
        orientations = {}
        for w in wanden:
            o = w["orientation"]
            if o not in orientations:
                orientations[o] = {"count": 0, "total_area_m2": 0}
            orientations[o]["count"] += 1
            orientations[o]["total_area_m2"] = round(
                orientations[o]["total_area_m2"] + w["area_m2"], 2
            )
        summary["totals"]["wanden_per_orientatie"] = orientations

    # Opslaan — converteer numpy types naar native Python
    def convert(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    result_path = output_dir / "analyse_resultaat.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=convert)

    print(f"\nResultaten opgeslagen in {result_path}")
    print(f"\nSamenvatting:")
    print(f"  Frames verwerkt: {len(frame_paths)}")
    print(f"  Vlakken gevonden: {len(all_surfaces)}")
    for label, data in summary["totals"].items():
        if isinstance(data, dict) and "total_area_m2" in data:
            print(f"  {label}: {data['count']}x, totaal {data['total_area_m2']} m²")

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Gebruik: python pipeline.py <video_path> [referentie_hoogte_m]")
        print("Voorbeeld: python pipeline.py ../input/kamer.mp4 2.10")
        sys.exit(1)

    video = sys.argv[1]
    ref_height = float(sys.argv[2]) if len(sys.argv) > 2 else 2.10

    run_pipeline(video, reference_height_m=ref_height)
