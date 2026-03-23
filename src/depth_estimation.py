"""Stap 2: Schat diepte per frame met Depth Anything V2."""

import cv2
import numpy as np
import torch
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image


class DepthEstimator:
    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"):
        """
        Laad Depth Anything V2 model.

        Metric-Indoor variant geeft absolute diepte in meters,
        geoptimaliseerd voor binnenruimtes.
        """
        print(f"Model laden: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model geladen.")

    @torch.no_grad()
    def estimate(self, image_path: str) -> np.ndarray:
        """
        Schat diepte voor één afbeelding.

        Returns:
            Dieptemap als numpy array in meters, zelfde resolutie als input.
        """
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (w, h)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        depth = outputs.predicted_depth.squeeze().cpu().numpy()

        # Resize naar originele resolutie
        depth = cv2.resize(depth, original_size, interpolation=cv2.INTER_LINEAR)

        return depth

    def process_frames(self, frame_dir: str, output_dir: str) -> list[Path]:
        """
        Verwerk alle frames in een directory.

        Returns:
            Lijst met paden naar opgeslagen dieptemaps (.npy bestanden).
        """
        frame_dir = Path(frame_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(frame_dir.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"Geen frames gevonden in {frame_dir}")

        print(f"\n{len(frames)} frames verwerken...")
        saved = []

        for i, frame_path in enumerate(frames):
            depth = self.estimate(str(frame_path))
            out_path = output_dir / f"{frame_path.stem}_depth.npy"
            np.save(str(out_path), depth)
            saved.append(out_path)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(frames)}] {frame_path.name} → "
                      f"diepte range: {depth.min():.2f}m - {depth.max():.2f}m")

        print(f"\n{len(saved)} dieptemaps opgeslagen in {output_dir}")
        return saved


if __name__ == "__main__":
    import sys

    frame_dir = sys.argv[1] if len(sys.argv) > 1 else "output/frames"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/depth"

    estimator = DepthEstimator()
    estimator.process_frames(frame_dir, out_dir)
