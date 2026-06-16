"""Sliding-window inference and metrics for patch-center CD models."""
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from sklearn import metrics


def center_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return logits at the spatial center.

    Args:
        logits: [B, C, H, W]
    Returns:
        [B, C]
    """
    h, w = logits.shape[-2:]
    return logits[:, :, h // 2, w // 2]


@torch.no_grad()
def predict_patch_centers(model, loader, device: torch.device) -> np.ndarray:
    """Predict one class per patch center from a loader."""
    model.eval()
    preds = []
    for x1, x2, _ in loader:
        x1 = x1.to(device=device, dtype=torch.float32)
        x2 = x2.to(device=device, dtype=torch.float32)
        outputs = model(x1, x2)
        pred = center_logits(outputs["final_logits"]).argmax(dim=1)
        preds.append(pred.cpu().numpy())
    if not preds:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(preds, axis=0).astype(np.int64)


def reconstruct_prediction_map(
    preds: np.ndarray,
    indices: np.ndarray,
    height: int,
    width: int,
    fill_value: int = -1,
) -> np.ndarray:
    """Fill a full-size map from flat pixel indices and center predictions."""
    pred_map = np.full((height, width), fill_value, dtype=np.int64)
    rows = indices // width
    cols = indices % width
    pred_map[rows, cols] = preds
    return pred_map


def compute_metrics(pred_map: np.ndarray, labels: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Compute OA, per-class accuracy, mean accuracy, and kappa."""
    if mask is None:
        mask = labels >= 0
    y_true = labels[mask].reshape(-1)
    y_pred = pred_map[mask].reshape(-1)
    valid = y_pred >= 0
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        return {"oa": 0.0, "aa": 0.0, "kappa": 0.0, "per_class_acc": {}}

    labels_present = sorted(np.unique(y_true).tolist())
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_present)
    denom = cm.sum(axis=1)
    per_class = {
        int(cls): float(cm[i, i] / denom[i]) if denom[i] > 0 else 0.0
        for i, cls in enumerate(labels_present)
    }
    return {
        "oa": float(metrics.accuracy_score(y_true, y_pred)),
        "aa": float(np.mean(list(per_class.values()))) if per_class else 0.0,
        "kappa": float(metrics.cohen_kappa_score(y_true, y_pred)),
        "per_class_acc": per_class,
    }


def _palette(num_classes: int) -> np.ndarray:
    base = np.asarray(
        [
            [0, 0, 0],
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
        ],
        dtype=np.uint8,
    )
    if num_classes <= base.shape[0]:
        return base[:num_classes]
    rng = np.random.default_rng(0)
    extra = rng.integers(0, 255, size=(num_classes - base.shape[0], 3), dtype=np.uint8)
    return np.concatenate([base, extra], axis=0)


def colorize_label_map(label_map: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """Colorize -1/0..K label maps. -1 is rendered white."""
    valid = label_map >= 0
    if num_classes is None:
        num_classes = int(label_map[valid].max()) + 1 if np.any(valid) else 1
    colors = _palette(num_classes)
    rgb = np.full((*label_map.shape, 3), 255, dtype=np.uint8)
    clipped = np.clip(label_map, 0, num_classes - 1)
    rgb[valid] = colors[clipped[valid]]
    return rgb


def save_label_png(label_map: np.ndarray, path: str, num_classes: Optional[int] = None) -> str:
    Image.fromarray(colorize_label_map(label_map, num_classes=num_classes)).save(path)
    return path


def save_prediction_outputs(
    pred_map: np.ndarray,
    output_dir: str,
    stem: str,
    labels: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, str]:
    """Save prediction map as .npy, .mat, and .png."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{stem}_pred.npy"
    mat_path = out_dir / f"{stem}_pred.mat"
    png_path = out_dir / f"{stem}_pred.png"
    np.save(npy_path, pred_map)
    sio.savemat(mat_path, {"prediction": pred_map})
    save_label_png(pred_map, str(png_path), num_classes=num_classes)
    paths = {"npy": str(npy_path), "mat": str(mat_path), "png": str(png_path)}
    if labels is not None:
        gt_path = out_dir / f"{stem}_gt.png"
        save_label_png(labels, str(gt_path), num_classes=num_classes)
        paths["gt_png"] = str(gt_path)
    return paths


def write_result_txt(path: str, metrics_dict: Dict[str, object], title: str = "") -> str:
    """Write OA/AA/KC and per-class accuracy."""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"OA={metrics_dict['oa']:.6f}")
    lines.append(f"AA={metrics_dict['aa']:.6f}")
    lines.append(f"KC={metrics_dict['kappa']:.6f}")
    lines.append("Per-class Acc:")
    for cls, acc in metrics_dict["per_class_acc"].items():
        lines.append(f"  class {cls}: {acc:.6f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
