"""
Evaluation script: load a trained UNet variant and compute metrics on BUSI.

Usage (after training in a notebook and saving weights):
  python evaluate.py --weights path/to/model.pth --data_dir path/to/Dataset_BUSI_with_GT --model unet

Models: unet | resnet_unet | attention_resnet_unet
You must implement get_model() to return the correct architecture so that
load_state_dict() matches. Keep architecture definitions in sync with the notebooks.
"""

import argparse
import torch
import numpy as np


def dice_coef(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    pred = (pred > 0.5).astype(np.float32).flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_coef(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    pred = (pred > 0.5).astype(np.float32).flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_recall_specificity(pred: np.ndarray, target: np.ndarray):
    pred = (pred > 0.5).astype(np.float32).flatten()
    target = target.flatten()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return float(prec), float(rec), float(spec)


def get_model(arch: str):
    """Return the model instance for the given architecture. Implement using your notebook definitions."""
    # TODO: import UNet / ResNetUNet / AttentionResNetUNet from a shared module (e.g. architectures.py)
    # and return the correct one so load_state_dict works.
    raise NotImplementedError("Implement get_model() using your notebook model classes.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate UNet on BUSI.")
    parser.add_argument("--weights", type=str, required=True, help="Path to saved model state dict.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to Dataset_BUSI_with_GT.")
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

    # TODO: build_data_dict + BUSIDataset from dataset.busi_loader;
    # run model on validation/test split and aggregate dice, iou, precision, recall, specificity.
    # Print or save results.
    print("Evaluation skeleton: connect get_model() and dataset, then run metrics above.")


if __name__ == "__main__":
    main()
