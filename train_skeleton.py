"""Runnable WPIP-Mamba-CD training skeleton with patch-center data loading.

Example:
    conda run -n MambaHSI_env python train_skeleton.py --dataset river --epochs 1 --device cpu
"""
import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import optim

from datasets.hsi_cd_dataset import build_hsi_cd_dataloaders
from losses.wpip_losses import WPIPLoss
from models.wpip_mamba import WPIPMambaCD
from utils.sliding_inference import (
    center_logits,
    compute_metrics,
    predict_patch_centers,
    reconstruct_prediction_map,
    save_prediction_outputs,
    write_result_txt,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train WPIP-Mamba-CD on patch-center HSI CD data.")
    parser.add_argument("--dataset", default="river", choices=["river", "china", "santa", "hermiston_usa"])
    parser.add_argument("--data-root", default="Comparison/CSANet-main/datasets")
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1331)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--train-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--inference-split", default="test", choices=["val", "test", "all"])
    parser.add_argument("--save-full-map", action="store_true")
    parser.add_argument("--no-balanced-train", action="store_true")
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--train-augment", action="store_true")
    parser.add_argument("--no-logit-calibration", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument(
        "--input-mode",
        default="dual",
        choices=["dual", "concat"],
        help="dual keeps separate T1/T2 branches; concat feeds T1||T2 as one HSI cube into the MambaHSI encoder.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def center_labels(labels: torch.Tensor) -> torch.Tensor:
    """Return center label from [B, H, W] center-only masks."""
    h, w = labels.shape[-2:]
    return labels[:, h // 2, w // 2]


def inverse_sqrt_class_weights(labels_np: np.ndarray, indices_np: np.ndarray, num_classes: int, power: float = 0.5) -> torch.Tensor:
    """Moderate class weights from center-pixel train labels."""
    flat = labels_np.reshape(-1)
    counts = np.bincount(flat[indices_np][flat[indices_np] >= 0], minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / np.power(counts, power)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


@torch.no_grad()
def update_center_prototypes(model: WPIPMambaCD, outputs: Dict[str, torch.Tensor], labels: torch.Tensor):
    """EMA-update prototypes from center semantic features and center labels."""
    feat = outputs["semantic_feature"]
    h, w = feat.shape[-2:]
    center_feat = feat[:, :, h // 2:h // 2 + 1, w // 2:w // 2 + 1].detach()
    center_lab = center_labels(labels).view(-1, 1, 1).detach()
    model.prototype_bank.update(center_feat, center_lab)


def train_one_epoch(model, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    running = 0.0
    seen = 0
    for x1, x2, labels in loader:
        x1 = x1.to(device=device, dtype=torch.float32)
        x2 = x2.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        outputs = model(x1, x2)
        loss_dict = criterion(outputs, labels)
        loss = loss_dict["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        update_center_prototypes(model, outputs, labels)

        batch = x1.shape[0]
        running += loss.item() * batch
        seen += batch
    return running / max(seen, 1)


@torch.no_grad()
def evaluate_center_oa(model, loader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    running = 0.0
    seen = 0
    correct = 0
    total = 0
    for x1, x2, labels in loader:
        x1 = x1.to(device=device, dtype=torch.float32)
        x2 = x2.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)
        outputs = model(x1, x2)
        loss = criterion(outputs, labels)["loss"]

        target = center_labels(labels)
        valid = target >= 0
        pred = center_logits(outputs["final_logits"]).argmax(dim=1)
        correct += (pred[valid] == target[valid]).sum().item()
        total += valid.sum().item()
        batch = x1.shape[0]
        running += loss.item() * batch
        seen += batch
    return {
        "loss": running / max(seen, 1),
        "center_oa": correct / max(total, 1),
    }


def run_sliding_inference(model, bundle, loader, split_name: str, output_dir: str, dataset_name: str, device: torch.device):
    preds = predict_patch_centers(model, loader, device)
    indices = bundle.splits[split_name]
    h, w = bundle.labels.shape
    pred_map = reconstruct_prediction_map(preds, indices, h, w)
    mask = np.zeros((h, w), dtype=bool)
    mask.reshape(-1)[indices] = True
    result_metrics = compute_metrics(pred_map, bundle.labels, mask=mask)
    num_classes = bundle.num_change_classes + 1
    paths = save_prediction_outputs(
        pred_map,
        output_dir,
        f"{dataset_name}_{split_name}",
        labels=bundle.labels,
        num_classes=num_classes,
    )
    paths["result_txt"] = write_result_txt(
        str(Path(output_dir) / f"{dataset_name}_{split_name}_result.txt"),
        result_metrics,
        title=f"{dataset_name} {split_name}",
    )
    return result_metrics, paths


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    bundle, loaders = build_hsi_cd_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        balanced_train=not args.no_balanced_train,
        train_augment=args.train_augment,
    )
    print(
        f"Loaded {args.dataset}: x={bundle.x1.shape}, classes=0..{bundle.num_change_classes}, "
        f"splits={{train:{len(bundle.splits['train'])}, val:{len(bundle.splits['val'])}, "
        f"test:{len(bundle.splits['test'])}}}"
    )

    model = WPIPMambaCD(
        in_channels=bundle.x1.shape[2] * (2 if args.input_mode == "concat" else 1),
        num_change_classes=bundle.num_change_classes,
        embed_dim=args.embed_dim,
        input_mode=args.input_mode,
        use_logit_calibration=not args.no_logit_calibration,
    ).to(device)
    semantic_weights = None
    binary_weights = None
    if not args.no_class_weight:
        semantic_weights = inverse_sqrt_class_weights(
            bundle.labels,
            bundle.splits["train"],
            num_classes=bundle.num_change_classes + 1,
            power=args.class_weight_power,
        )
        binary_train = (bundle.labels.reshape(-1)[bundle.splits["train"]] > 0).astype(np.int64)
        bin_counts = np.bincount(binary_train, minlength=2).astype(np.float64)
        bin_counts = np.maximum(bin_counts, 1.0)
        bin_w = 1.0 / np.power(bin_counts, args.class_weight_power)
        binary_weights = torch.as_tensor(bin_w / bin_w.mean(), dtype=torch.float32)
        print(f"semantic_class_weights={semantic_weights.tolist()}")
        print(f"binary_class_weights={binary_weights.tolist()}")
    criterion = WPIPLoss(lambda_pseudo=0.1, semantic_class_weights=semantic_weights, binary_class_weights=binary_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_stats = evaluate_center_oa(model, loaders["val"], criterion, device)
        is_best = val_stats["center_oa"] >= best_val
        best_val = max(best_val, val_stats["center_oa"])
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_center_oa={val_stats['center_oa']:.4f}"
        )
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "num_change_classes": bundle.num_change_classes,
            "in_channels": bundle.x1.shape[2] * (2 if args.input_mode == "concat" else 1),
            "input_mode": args.input_mode,
            "best_val_center_oa": best_val,
        }
        torch.save(checkpoint, output_dir / f"{args.dataset}_latest.pt")
        if is_best:
            torch.save(checkpoint, output_dir / f"{args.dataset}_best.pt")

    if not args.skip_inference:
        best_path = output_dir / f"{args.dataset}_best.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
        test_metrics, paths = run_sliding_inference(
            model,
            bundle,
            loaders[args.inference_split],
            split_name=args.inference_split,
            output_dir=str(output_dir),
            dataset_name=args.dataset,
            device=device,
        )
        summary = {"best_val_center_oa": best_val, args.inference_split: test_metrics, "outputs": paths}
        if args.save_full_map and args.inference_split != "all":
            all_metrics, all_paths = run_sliding_inference(
                model,
                bundle,
                loaders["all"],
                split_name="all",
                output_dir=str(output_dir),
                dataset_name=args.dataset,
                device=device,
            )
            summary["all"] = all_metrics
            summary["full_map_outputs"] = all_paths
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
