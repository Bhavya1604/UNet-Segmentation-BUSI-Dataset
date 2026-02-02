"""
Inference script: load a trained UNet variant and predict masks for images.

Usage:
  python inference.py --weights path/to/model.pth --input path/to/image.png --output path/to/pred_mask.png --model resnet_unet

For batch: use --input as a directory and --output as directory (implement loop over images).
"""

import argparse
import os
import numpy as np
from PIL import Image
import torch


def get_model(arch: str):
    """Return the model instance. Implement using your notebook model definitions."""
    # TODO: import from architectures.py or notebooks and return correct model.
    raise NotImplementedError("Implement get_model() so architecture matches saved weights.")


def load_image(path: str, size=(256, 256)):
    """Load grayscale image, resize, normalize to [0,1], return tensor [1, 1, H, W]."""
    img = Image.open(path).convert("L")
    img = img.resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def save_mask(arr: np.ndarray, path: str):
    """Save binary mask as image (0 or 255)."""
    out = ((arr > 0.5).astype(np.uint8)) * 255
    Image.fromarray(out).save(path)


def main():
    parser = argparse.ArgumentParser(description="Run UNet inference on an image.")
    parser.add_argument("--weights", type=str, required=True, help="Path to model state dict.")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory.")
    parser.add_argument("--output", type=str, required=True, help="Output mask path or directory.")
    parser.add_argument("--model", type=str, choices=["unet", "resnet_unet", "attention_resnet_unet"], default="resnet_unet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = get_model(args.model)
    state = torch.load(args.weights, map_location=args.device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(args.device)
    model.eval()

    if os.path.isfile(args.input):
        x = load_image(args.input).to(args.device)
        with torch.no_grad():
            logits = model(x)
        pred = torch.sigmoid(logits).squeeze().cpu().numpy()
        save_mask(pred, args.output)
        print("Saved mask to", args.output)
    else:
        # TODO: loop over images in directory and save to output directory.
        print("Batch inference: implement directory loop (--input dir, --output dir).")


if __name__ == "__main__":
    main()
