"""Stap 3: Detecteer vlakken (wanden, vloer, plafond) uit dieptemaps."""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN


@dataclass
class Surface:
    """Een gedetecteerd vlak in de ruimte."""
    label: str  # wand, vloer, plafond, raam, deur
    normal: np.ndarray  # normaalvector (3D)
    area_m2: float  # oppervlakte in m²
    orientation: str  # N, Z, O, W, horizontaal
    center: np.ndarray  # centerpunt in 3D
    dimensions: dict = field(default_factory=dict)  # breedte, hoogte in meters
    pixels: np.ndarray | None = None  # pixel mask voor visualisatie


def depth_to_3d(depth_map: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Converteer dieptemap naar 3D punten.

    Args:
        depth_map: Dieptemap in meters (H x W).
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point.

    Returns:
        3D punten als (H x W x 3) array.
    """
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1)


def compute_normals(points_3d: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Bereken oppervlaktenormalen via cross product van naburige punten."""
    padded = np.pad(points_3d, ((kernel_size, kernel_size), (kernel_size, kernel_size), (0, 0)), mode='edge')

    # Vectors in twee richtingen
    dv = padded[2*kernel_size:, kernel_size:-kernel_size] - padded[:-2*kernel_size, kernel_size:-kernel_size]
    du = padded[kernel_size:-kernel_size, 2*kernel_size:] - padded[kernel_size:-kernel_size, :-2*kernel_size]

    normals = np.cross(du, dv)

    # Normaliseer
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    normals = normals / norms

    return normals


def classify_surface(normal: np.ndarray) -> str:
    """Classificeer een vlak op basis van zijn normaalvector."""
    # Y-as is verticaal in camera-coördinaten
    abs_normal = np.abs(normal)

    # Verticaal vlak (normaal hoofdzakelijk horizontaal) → wand
    if abs_normal[1] < 0.4:
        return "wand"

    # Horizontaal vlak, normaal naar beneden → vloer
    if normal[1] > 0.4:
        return "vloer"

    # Horizontaal vlak, normaal naar boven → plafond
    if normal[1] < -0.4:
        return "plafond"

    return "onbekend"


def get_orientation(normal: np.ndarray) -> str:
    """Bepaal de oriëntatie (windrichting) van een verticaal vlak."""
    # Negeer verticale component
    horizontal = np.array([normal[0], normal[2]])
    if np.linalg.norm(horizontal) < 0.1:
        return "horizontaal"

    angle = np.degrees(np.arctan2(horizontal[0], horizontal[1])) % 360

    if 315 <= angle or angle < 45:
        return "N"
    elif 45 <= angle < 135:
        return "O"
    elif 135 <= angle < 225:
        return "Z"
    else:
        return "W"


def detect_surfaces(
    depth_map: np.ndarray,
    rgb_frame: np.ndarray,
    fx: float | None = None,
    fy: float | None = None,
    min_surface_area: float = 0.5,
    scale_factor: float = 1.0,
) -> list[Surface]:
    """
    Detecteer vlakken in een dieptemap.

    Args:
        depth_map: Dieptemap in meters.
        rgb_frame: Bijbehorend RGB frame (voor kleuranalyse).
        fx, fy: Focal lengths. Als None, worden ze geschat uit beeldgrootte.
        min_surface_area: Minimale oppervlakte in m² om als vlak te tellen.
        scale_factor: Schaalcorrectie (bijv. gemeten_maat / geschatte_maat).

    Returns:
        Lijst met gedetecteerde Surfaces.
    """
    h, w = depth_map.shape
    depth_scaled = depth_map * scale_factor

    # Schat camera intrinsics als niet gegeven (typische smartphone FOV ~60-70°)
    if fx is None:
        fx = w * 0.8
    if fy is None:
        fy = h * 0.8
    cx, cy = w / 2, h / 2

    # Stap 1: Diepte → 3D punten
    points_3d = depth_to_3d(depth_scaled, fx, fy, cx, cy)

    # Stap 2: Bereken normalen
    normals = compute_normals(points_3d)

    # Stap 3: Classificeer elke pixel
    labels_map = np.full((h, w), -1, dtype=int)
    surface_types = np.empty((h, w), dtype=object)

    for v in range(h):
        for u in range(w):
            if depth_scaled[v, u] > 0.1:  # filter noise
                surface_types[v, u] = classify_surface(normals[v, u])

    # Stap 4: Cluster aaneengesloten vlakken van hetzelfde type
    surfaces = []

    for surface_type in ["wand", "vloer", "plafond"]:
        mask = surface_types == surface_type
        if mask.sum() < 100:
            continue

        # Gebruik connected components voor clustering
        mask_uint8 = mask.astype(np.uint8) * 255
        num_labels, label_map = cv2.connectedComponents(mask_uint8)

        for label_id in range(1, num_labels):
            region = label_map == label_id
            pixel_count = region.sum()

            if pixel_count < 100:
                continue

            # Bereken oppervlakte in m²
            region_points = points_3d[region]
            region_normals = normals[region]
            mean_normal = region_normals.mean(axis=0)
            mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)

            # Schat oppervlakte via pixel-diepte sampling
            region_depths = depth_scaled[region]
            mean_depth = region_depths.mean()
            pixel_area = (mean_depth / fx) * (mean_depth / fy)
            area_m2 = pixel_count * pixel_area

            if area_m2 < min_surface_area:
                continue

            # Bounding box voor dimensies
            ys, xs = np.where(region)
            bbox_h = (ys.max() - ys.min()) * mean_depth / fy
            bbox_w = (xs.max() - xs.min()) * mean_depth / fx

            center = region_points.mean(axis=0)
            orientation = get_orientation(mean_normal) if surface_type == "wand" else "horizontaal"

            surfaces.append(Surface(
                label=surface_type,
                normal=mean_normal,
                area_m2=round(area_m2, 2),
                orientation=orientation,
                center=center,
                dimensions={"breedte_m": round(bbox_w, 2), "hoogte_m": round(bbox_h, 2)},
                pixels=region,
            ))

    # Stap 5: Detecteer deuropeningen binnen wand-regio's
    # Een deuropening is een rechthoekig gebied binnen een wand
    # waar de diepte significant groter is (je kijkt erdoorheen)
    surfaces = _detect_openings(surfaces, depth_scaled, points_3d, fx, fy, cx, cy)

    return surfaces


def _detect_openings(
    surfaces: list[Surface],
    depth: np.ndarray,
    points_3d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_jump_threshold: float = 0.3,
) -> list[Surface]:
    """
    Detecteer deur/raam-openingen binnen wand-regio's.

    Zoekt naar gebieden in wanden waar de diepte ineens veel groter is
    (= je kijkt door een opening naar de ruimte erachter).
    """
    h, w = depth.shape
    new_surfaces = []

    for surface in surfaces:
        if surface.label != "wand" or surface.pixels is None:
            new_surfaces.append(surface)
            continue

        wall_mask = surface.pixels
        wall_depths = depth.copy()
        wall_depths[~wall_mask] = 0

        # Mediaan diepte van de wand
        wall_depth_values = depth[wall_mask]
        median_depth = np.median(wall_depth_values)

        # Pixels die significant dieper zijn dan de wand = opening
        opening_mask = wall_mask & (depth > median_depth + depth_jump_threshold)

        if opening_mask.sum() < 50:
            new_surfaces.append(surface)
            continue

        # Morphological cleanup van de opening-mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening_clean = cv2.morphologyEx(
            opening_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel
        )
        opening_clean = cv2.morphologyEx(opening_clean, cv2.MORPH_OPEN, kernel)

        # Vind rechthoekige contouren (deuren zijn typisch rechthoekig)
        contours, _ = cv2.findContours(opening_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        remaining_wall = wall_mask.copy()

        for contour in contours:
            area_px = cv2.contourArea(contour)
            if area_px < 200:
                continue

            # Bounding rect
            x, y, bw, bh = cv2.boundingRect(contour)

            # Aspect ratio check: deur is hoger dan breed (ratio > 1.2)
            # Raam is breder dan hoog of kleiner
            aspect = bh / max(bw, 1)

            # Maak een mask voor deze opening
            opening_region = np.zeros((h, w), dtype=bool)
            cv2.drawContours(opening_region.astype(np.uint8), [contour], -1, 1, -1)
            opening_region = opening_region.astype(bool) | (
                (np.arange(h)[:, None] >= y) & (np.arange(h)[:, None] < y + bh) &
                (np.arange(w)[None, :] >= x) & (np.arange(w)[None, :] < x + bw) &
                wall_mask
            )
            # Verfijn: alleen pixels binnen de contour
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            opening_region = (contour_mask > 0) & wall_mask

            if opening_region.sum() < 50:
                continue

            # Bereken afmetingen
            region_depths = depth[opening_region]
            mean_depth = np.median(region_depths)
            opening_h = bh * mean_depth / fy
            opening_w = bw * mean_depth / fx
            area_m2 = opening_region.sum() * (mean_depth / fx) * (mean_depth / fy)

            # Classificeer: deur als hoog + begint onderaan, anders raam
            bottom_of_opening = y + bh
            frame_bottom = h
            touches_bottom = bottom_of_opening > frame_bottom * 0.8

            if aspect > 1.2 and touches_bottom:
                label = "deur"
            elif aspect > 1.2:
                label = "deur"  # hoge opening, waarschijnlijk deur
            else:
                label = "raam"

            center = points_3d[opening_region].mean(axis=0) if opening_region.any() else surface.center

            new_surfaces.append(Surface(
                label=label,
                normal=surface.normal,
                area_m2=round(area_m2, 2),
                orientation=surface.orientation,
                center=center,
                dimensions={"breedte_m": round(opening_w, 2), "hoogte_m": round(opening_h, 2)},
                pixels=opening_region,
            ))

            # Verwijder opening-pixels uit de wand
            remaining_wall = remaining_wall & ~opening_region

        # Update de wand (zonder de opening-pixels)
        if remaining_wall.sum() > 100:
            wall_depths_remaining = depth[remaining_wall]
            mean_depth = np.median(wall_depths_remaining)
            pixel_area = (mean_depth / fx) * (mean_depth / fy)
            remaining_area = remaining_wall.sum() * pixel_area

            ys, xs = np.where(remaining_wall)
            bbox_h = (ys.max() - ys.min()) * mean_depth / fy
            bbox_w = (xs.max() - xs.min()) * mean_depth / fx

            new_surfaces.append(Surface(
                label="wand",
                normal=surface.normal,
                area_m2=round(remaining_area, 2),
                orientation=surface.orientation,
                center=points_3d[remaining_wall].mean(axis=0),
                dimensions={"breedte_m": round(bbox_w, 2), "hoogte_m": round(bbox_h, 2)},
                pixels=remaining_wall,
            ))
        else:
            new_surfaces.append(surface)

    return new_surfaces


if __name__ == "__main__":
    import sys
    import json

    depth_path = sys.argv[1] if len(sys.argv) > 1 else "output/depth/frame_000000_depth.npy"
    frame_path = sys.argv[2] if len(sys.argv) > 2 else "output/frames/frame_000000.jpg"

    depth = np.load(depth_path)
    frame = cv2.imread(frame_path)

    surfaces = detect_surfaces(depth, frame)

    print(f"\n{len(surfaces)} vlakken gedetecteerd:\n")
    for s in surfaces:
        print(f"  {s.label:10s} | {s.area_m2:6.1f} m² | {s.orientation:4s} | "
              f"{s.dimensions.get('breedte_m', 0):.1f}m x {s.dimensions.get('hoogte_m', 0):.1f}m")
