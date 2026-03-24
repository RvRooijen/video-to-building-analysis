"""Stap 3b: Semantic segmentation met OneFormer (ADE20K)."""

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation


# ADE20K labels die relevant zijn voor gebouwanalyse
BUILDING_LABELS = {
    "wall": "wand",
    "floor": "vloer",
    "ceiling": "plafond",
    "door": "deur",
    "window": "raam",
    "windowpane": "raam",
    "column": "kolom",
    "stairs": "trap",
    "railing": "railing",
    "beam": "balk",
    "pipe": "leiding",
    "vent": "ventilatie",
}

# Alle ADE20K labels (150 klassen) — we filteren op de relevante
# Volledige lijst: https://huggingface.co/datasets/huggingface/label-files/blob/main/ade20k-id2label.json


class SemanticSegmenter:
    def __init__(self, model_name: str = "shi-labs/oneformer_ade20k_swin_large"):
        print(f"Model laden: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Haal label mapping op
        self.id2label = self.model.config.id2label
        print(f"Model geladen. {len(self.id2label)} klassen beschikbaar.")

        # Log welke gebouw-relevante labels beschikbaar zijn
        # Strip whitespace uit labels (sommige modellen hebben trailing spaces)
        self.id2label = {k: v.strip() for k, v in self.id2label.items()}

        relevant = {k: v for k, v in self.id2label.items()
                    if v.lower() in BUILDING_LABELS}
        print(f"Relevante gebouwlabels: {relevant}")

    @torch.no_grad()
    def segment(self, image_path: str) -> tuple[np.ndarray, dict]:
        """
        Semantic segmentation op één afbeelding.

        Returns:
            segmentation_map: (H, W) array met klasse-IDs
            label_info: dict met {klasse_id: {"label_en": ..., "label_nl": ..., "pixel_count": ...}}
        """
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (w, h)

        inputs = self.processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        seg_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0].cpu().numpy()

        # Verzamel info per gedetecteerd label
        label_info = {}
        unique_ids = np.unique(seg_map)
        for uid in unique_ids:
            uid = int(uid)
            en_label = self.id2label.get(uid, "unknown")
            nl_label = BUILDING_LABELS.get(en_label.lower(), en_label)
            pixel_count = int((seg_map == uid).sum())
            label_info[uid] = {
                "label_en": en_label,
                "label_nl": nl_label,
                "pixel_count": pixel_count,
                "is_building_element": en_label.lower() in BUILDING_LABELS,
            }

        return seg_map, label_info

    def process_frames(self, frame_dir: str, output_dir: str) -> list[Path]:
        """Verwerk alle frames."""
        frame_dir = Path(frame_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(frame_dir.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"Geen frames gevonden in {frame_dir}")

        print(f"\n{len(frames)} frames segmenteren...")
        saved = []

        for i, frame_path in enumerate(frames):
            seg_map, label_info = self.segment(str(frame_path))

            out_path = output_dir / f"{frame_path.stem}_seg.npy"
            np.save(str(out_path), seg_map)
            saved.append(out_path)

            if (i + 1) % 5 == 0 or i == 0:
                building_elements = [v for v in label_info.values() if v["is_building_element"]]
                labels_found = [v["label_nl"] for v in building_elements]
                print(f"  [{i+1}/{len(frames)}] {frame_path.name} → {', '.join(labels_found)}")

        print(f"\n{len(saved)} segmentatie-maps opgeslagen in {output_dir}")
        return saved


if __name__ == "__main__":
    import sys

    frame_dir = sys.argv[1] if len(sys.argv) > 1 else "output/frames"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/segmentation"

    segmenter = SemanticSegmenter()
    segmenter.process_frames(frame_dir, out_dir)
