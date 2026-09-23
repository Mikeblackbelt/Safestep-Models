"""
Generates synthetic sample images and text datasets for testing the SafeStep API.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def generate_test_assets(output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Authentic-like image (natural gradients and noise)
    img_auth = Image.new("RGB", (256, 256), color=(135, 206, 235))
    draw = ImageDraw.Draw(img_auth)
    draw.rectangle([50, 100, 200, 220], fill=(34, 139, 34))
    draw.ellipse([80, 40, 140, 100], fill=(255, 255, 0))
    auth_path = output_dir / "sample_authentic.png"
    img_auth.save(auth_path)

    # 2. AI-like image (checkerboard / high frequency pattern)
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            if (i // 16 + j // 16) % 2 == 0:
                arr[i, j] = [200, 50, 120]
            else:
                arr[i, j] = [50, 200, 180]
    img_ai = Image.fromarray(arr)
    ai_path = output_dir / "sample_synthetic.png"
    img_ai.save(ai_path)

    print(f"Generated test assets at {output_dir}:")
    print(f" - {auth_path.name}")
    print(f" - {ai_path.name}")
    return auth_path, ai_path


if __name__ == "__main__":
    generate_test_assets(Path("tests/samples"))
