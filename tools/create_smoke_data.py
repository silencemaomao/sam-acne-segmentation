from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sample(image_path: Path, label_path: Path, seed: int, positive: bool) -> None:
    rng = np.random.default_rng(seed)
    pixels = np.full((64, 64, 3), [188, 142, 122], dtype=np.int16)
    pixels += rng.normal(0, 6, pixels.shape).astype(np.int16)
    image, label = Image.fromarray(np.uint8(np.clip(pixels, 0, 255))), Image.new("L", (64, 64), 0)
    if positive:
        x, y, radius = int(rng.integers(14, 50)), int(rng.integers(14, 50)), int(rng.integers(3, 7))
        ImageDraw.Draw(image).ellipse((x - radius, y - radius, x + radius, y + radius), fill=(205, 65, 65))
        draw = ImageDraw.Draw(label)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        draw.rectangle((0, 0, 5, 63), fill=128)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    label.save(label_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("smoke_data"))
    args = parser.parse_args()
    for split, count, offset in (("train", 8, 0), ("val", 4, 100), ("test", 4, 200)):
        for index in range(count):
            sample(
                args.output_dir / split / "images" / f"sample_{index:03d}.jpg",
                args.output_dir / split / "labels" / f"sample_{index:03d}.png",
                offset + index,
                index % 2 == 0,
            )
    print(f"Created smoke data under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
