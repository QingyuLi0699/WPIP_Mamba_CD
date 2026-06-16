"""Tune binary-guided two-stage decision for a trained WPIP-Mamba-CD checkpoint.

The model is unchanged. This script uses the binary head for the decision it was
designed for: first separate no-change/change, then classify semantic change
classes only inside predicted change pixels.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from datasets.hsi_cd_dataset import build_hsi_cd_dataloaders
from models.wpip_mamba import WPIPMambaCD
from train_skeleton import center_labels
from utils.sliding_inference import (
    compute_metrics,
    reconstruct_prediction_map,
    save_prediction_outputs,
    write_result_txt,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune binary-gated inference for WPIP-Mamba-CD.")
    parser.add_argument("--checkpoint", default="outputs_hermiston_aug_p025_tr20/hermiston_usa_best.pt")
    parser.add_argument("--dataset", default="hermiston_usa")
    parser.add_argument("--data-root", default="dataset/hermiston_USA")
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1331)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs_hermiston_aug_p025_tr20_binary_gate")
    return parser.parse_args()


def load_model(checkpoint_path: str, bundle, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    input_mode = ckpt.get("input_mode", ckpt_args.get("input_mode", "dual"))
    model = WPIPMambaCD(
        in_channels=ckpt.get("in_channels", bundle.x1.shape[2] * (2 if input_mode == "concat" else 1)),
        num_change_classes=ckpt.get("num_change_classes", bundle.num_change_classes),
        embed_dim=ckpt_args.get("embed_dim", 128),
        input_mode=input_mode,
        use_logit_calibration=not ckpt_args.get("no_logit_calibration", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def collect_split(model, loader, device: torch.device):
    semantic_logits = []
    binary_prob = []
    labels = []
    binary_pred = []
    for x1, x2, y in loader:
        x1 = x1.to(device=device, dtype=torch.float32)
        x2 = x2.to(device=device, dtype=torch.float32)
        outputs = model(x1, x2)
        h, w = outputs["final_logits"].shape[-2:]
        logits = outputs["final_logits"][:, :, h // 2, w // 2]
        bin_logits = outputs["binary_logits"][:, :, h // 2, w // 2]
        p_change = torch.softmax(bin_logits, dim=1)[:, 1]
        semantic_logits.append(logits.cpu())
        binary_prob.append(p_change.cpu())
        binary_pred.append((p_change.cpu() > 0.5).long())
        labels.append(center_labels(y).long())
    return {
        "semantic_logits": torch.cat(semantic_logits, dim=0),
        "binary_prob": torch.cat(binary_prob, dim=0),
        "binary_pred": torch.cat(binary_pred, dim=0),
        "labels": torch.cat(labels, dim=0),
    }


def gated_predict(data, threshold: float, class1_bias: float = 0.0):
    logits = data["semantic_logits"].clone()
    logits[:, 1] += class1_bias
    change_cls = logits[:, 1:].argmax(dim=1) + 1
    pred = torch.where(data["binary_prob"] >= threshold, change_cls, torch.zeros_like(change_cls))
    return pred.numpy().astype(np.int64)


def vector_metrics(pred: np.ndarray, labels: np.ndarray):
    labels = labels.astype(np.int64)
    cls = sorted(np.unique(labels[labels >= 0]).tolist())
    per = {}
    for c in cls:
        mask = labels == c
        per[c] = float((pred[mask] == labels[mask]).mean()) if mask.any() else 0.0
    oa = float((pred == labels).mean())
    aa = float(np.mean(list(per.values())))
    return oa, aa, per


def score_candidate(metrics, targets):
    oa, aa, per = metrics
    deficits = 0.0
    for cls, target in targets.items():
        deficits += max(0.0, target - per.get(cls, 0.0)) * 10.0
    # Strongly prefer satisfying class 0/1, then AA, then OA.
    return aa + 0.25 * oa - deficits


def main():
    args = parse_args()
    device = torch.device(args.device)
    bundle, loaders = build_hsi_cd_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        balanced_train=False,
        train_augment=False,
    )
    model = load_model(args.checkpoint, bundle, device)

    val = collect_split(model, loaders["val"], device)
    y_val = val["labels"].numpy()
    baseline = val["semantic_logits"].argmax(dim=1).numpy()
    binary = val["binary_pred"].numpy()
    print("val semantic baseline", vector_metrics(baseline, y_val))
    print("val binary head", vector_metrics(binary, (y_val > 0).astype(np.int64)))

    targets = {0: 0.97, 1: 0.88}
    candidates = []
    for threshold in np.linspace(0.05, 0.95, 91):
        for class1_bias in np.linspace(-1.0, 1.5, 51):
            pred = gated_predict(val, float(threshold), float(class1_bias))
            m = vector_metrics(pred, y_val)
            candidates.append((score_candidate(m, targets), float(threshold), float(class1_bias), m))
    candidates.sort(key=lambda x: x[0], reverse=True)
    print("top validation candidates")
    for item in candidates[:10]:
        _, threshold, class1_bias, m = item
        print(f"threshold={threshold:.3f} class1_bias={class1_bias:.3f} metrics={m}")

    best = candidates[0]
    threshold, class1_bias = best[1], best[2]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_config.txt").write_text(
        f"checkpoint={args.checkpoint}\nthreshold={threshold:.6f}\nclass1_bias={class1_bias:.6f}\n",
        encoding="utf-8",
    )

    for split in ("test", "all"):
        data = collect_split(model, loaders[split], device)
        pred = gated_predict(data, threshold, class1_bias)
        labels = data["labels"].numpy()
        print(split, "vector", vector_metrics(pred, labels))
        indices = bundle.splits[split]
        pred_map = reconstruct_prediction_map(pred, indices, *bundle.labels.shape)
        mask = np.zeros(bundle.labels.shape, dtype=bool)
        mask.reshape(-1)[indices] = True
        metrics = compute_metrics(pred_map, bundle.labels, mask=mask)
        paths = save_prediction_outputs(
            pred_map,
            str(out_dir),
            f"{args.dataset}_{split}",
            labels=bundle.labels,
            num_classes=bundle.num_change_classes + 1,
        )
        paths["result_txt"] = write_result_txt(
            str(out_dir / f"{args.dataset}_{split}_result.txt"),
            metrics,
            title=f"{args.dataset} {split} binary-gated",
        )
        print(split, metrics)
        print(paths)


if __name__ == "__main__":
    main()
