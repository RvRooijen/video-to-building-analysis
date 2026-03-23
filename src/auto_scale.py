"""Automatische schaalcorrectie op basis van een referentie-object (bijv. deur)."""

import numpy as np
from pathlib import Path


# Bekende maten van referentie-objecten (in meters)
REFERENCE_OBJECTS = {
    "deur": {"hoogte_m": 2.05, "breedte_m": 0.83},
    "a4": {"hoogte_m": 0.297, "breedte_m": 0.210},
}


def compute_scale_from_reference(
    seg_map: np.ndarray,
    depth_map: np.ndarray,
    id2label: dict,
    reference_type: str = "deur",
    fy: float | None = None,
    fx: float | None = None,
) -> tuple[float, dict]:
    """
    Bereken schaalfactor op basis van een referentie-object.

    Zoekt het referentie-object in de segmentatie-map, meet de geschatte
    afmetingen via de dieptemap, en vergelijkt met de bekende maten.

    Returns:
        scale_factor: correctiefactor (vermenigvuldig met geschatte maten)
        info: dict met details over de berekening
    """
    h, w = seg_map.shape

    if fy is None:
        fy = h * 0.8
    if fx is None:
        fx = w * 0.8

    known = REFERENCE_OBJECTS[reference_type]

    # Zoek het label-ID voor het referentie-object
    label_map = {v.lower(): k for k, v in id2label.items()}

    # Map NL referentie-type naar EN label
    nl_to_en = {"deur": "door", "a4": None}
    en_label = nl_to_en.get(reference_type)

    if en_label is None or en_label not in label_map:
        return 1.0, {"error": f"Referentie-object '{reference_type}' niet gevonden in model labels"}

    ref_id = label_map[en_label]
    mask = seg_map == ref_id

    if mask.sum() < 100:
        return 1.0, {"error": f"Referentie-object '{reference_type}' niet gedetecteerd in dit frame"}

    # Bereken geschatte afmetingen
    ys, xs = np.where(mask)
    region_depths = depth_map[mask]
    median_depth = np.median(region_depths)

    estimated_height = (ys.max() - ys.min()) * median_depth / fy
    estimated_width = (xs.max() - xs.min()) * median_depth / fx

    # Bereken schaalfactor op basis van hoogte (meest betrouwbaar)
    scale_height = known["hoogte_m"] / estimated_height if estimated_height > 0 else 1.0
    scale_width = known["breedte_m"] / estimated_width if estimated_width > 0 else 1.0

    # Gebruik gemiddelde van beide, gewogen naar hoogte (meer pixels, betrouwbaarder)
    scale_factor = scale_height * 0.7 + scale_width * 0.3

    info = {
        "reference_type": reference_type,
        "known_height_m": known["hoogte_m"],
        "known_width_m": known["breedte_m"],
        "estimated_height_m": round(estimated_height, 3),
        "estimated_width_m": round(estimated_width, 3),
        "scale_from_height": round(scale_height, 4),
        "scale_from_width": round(scale_width, 4),
        "scale_factor": round(scale_factor, 4),
        "median_depth_m": round(float(median_depth), 3),
    }

    return scale_factor, info


def compute_scale_from_all_frames(
    seg_dir: str,
    depth_dir: str,
    id2label: dict,
    reference_type: str = "deur",
) -> tuple[float, dict]:
    """
    Bereken de beste schaalfactor over alle frames.

    Pakt frames waar het referentie-object het grootst/duidelijkst is.
    """
    seg_dir = Path(seg_dir)
    depth_dir = Path(depth_dir)

    seg_files = sorted(seg_dir.glob("*.npy"))
    depth_files = sorted(depth_dir.glob("*.npy"))

    all_scales = []
    best_info = None
    best_pixel_count = 0

    for seg_path, depth_path in zip(seg_files, depth_files):
        seg_map = np.load(str(seg_path))
        depth_map = np.load(str(depth_path))

        scale, info = compute_scale_from_reference(
            seg_map, depth_map, id2label, reference_type
        )

        if "error" in info:
            continue

        # Zoek het label-ID
        label_map = {v.lower(): k for k, v in id2label.items()}
        nl_to_en = {"deur": "door"}
        en_label = nl_to_en.get(reference_type)
        ref_id = label_map[en_label]
        pixel_count = (seg_map == ref_id).sum()

        all_scales.append(scale)

        if pixel_count > best_pixel_count:
            best_pixel_count = pixel_count
            best_info = info
            best_info["frame"] = seg_path.name

    if not all_scales:
        print(f"  Referentie-object '{reference_type}' niet gevonden in enig frame!")
        return 1.0, {}

    # Gebruik mediaan om outliers te filteren
    median_scale = float(np.median(all_scales))

    print(f"  Schaalfactor berekend uit {len(all_scales)} frames")
    print(f"  Mediaan schaalfactor: {median_scale:.4f}")
    print(f"  Beste frame: {best_info.get('frame', '?')}")
    print(f"  Geschatte deur: {best_info['estimated_height_m']}m x {best_info['estimated_width_m']}m")
    print(f"  Bekende deur:   {best_info['known_height_m']}m x {best_info['known_width_m']}m")
    print(f"  → Correctie: x{median_scale:.3f}")

    return median_scale, best_info


if __name__ == "__main__":
    import json
    from semantic_segmentation import SemanticSegmenter

    segmenter = SemanticSegmenter()
    scale, info = compute_scale_from_all_frames(
        "output/segmentation", "output/depth", segmenter.id2label, "deur"
    )
    print(f"\nResultaat: schaalfactor = {scale:.4f}")
    print(json.dumps(info, indent=2))
