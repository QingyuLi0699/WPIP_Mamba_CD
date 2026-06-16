"""Tune threshold for adaptive semantic-binary gate predictions."""
import argparse
from pathlib import Path

import numpy as np
import torch

from datasets.hsi_cd_dataset import build_hsi_cd_dataloaders
from models.wpip_mamba import WPIPMambaCD
from train_skeleton import center_labels
from utils.sliding_inference import compute_metrics, reconstruct_prediction_map, save_prediction_outputs, write_result_txt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs_hermiston_adaptive_gate_e20/hermiston_usa_best.pt")
    parser.add_argument("--output-dir", default="outputs_hermiston_adaptive_gate_e20_tuned")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def load_model(checkpoint_path, bundle, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    input_mode = ckpt.get("input_mode", ckpt_args.get("input_mode", "dual"))
    model = WPIPMambaCD(
        in_channels=ckpt.get("in_channels", bundle.x1.shape[2]),
        num_change_classes=ckpt.get("num_change_classes", bundle.num_change_classes),
        embed_dim=ckpt_args.get("embed_dim", 128),
        input_mode=input_mode,
        use_logit_calibration=not ckpt_args.get("no_logit_calibration", False),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def collect(model, loader, device):
    probs, preds, labels = [], [], []
    for x1, x2, y in loader:
        x1 = x1.to(device=device, dtype=torch.float32)
        x2 = x2.to(device=device, dtype=torch.float32)
        out = model(x1, x2)
        h, w = out["final_logits"].shape[-2:]
        final_center = out["final_logits"][:, :, h // 2, w // 2]
        sem_pred = final_center[:, 1:].argmax(dim=1).cpu().numpy() + 1
        gh, gw = out["adaptive_change_prob"].shape[-2:]
        prob = out["adaptive_change_prob"][:, 0, gh // 2, gw // 2].cpu().numpy()
        probs.append(prob)
        preds.append(sem_pred.astype(np.int64))
        labels.append(center_labels(y).numpy().astype(np.int64))
    return np.concatenate(probs), np.concatenate(preds), np.concatenate(labels)


def vector_metrics(pred, labels):
    per = {}
    for c in sorted(np.unique(labels).tolist()):
        m = labels == c
        per[int(c)] = float((pred[m] == labels[m]).mean())
    return float((pred == labels).mean()), float(np.mean(list(per.values()))), per


def apply_threshold(prob, sem_pred, threshold):
    return np.where(prob >= threshold, sem_pred, 0).astype(np.int64)


def score(metrics):
    oa, aa, per = metrics
    ref = {0: 0.97, 1: 0.88, 2: 0.90, 3: 0.93, 4: 0.82, 5: 0.90, 6: 0.82}
    deficit = sum((4.0 if c in (0, 1) else 2.0) * max(0.0, t - per.get(c, 0.0)) for c, t in ref.items())
    return aa + 0.3 * oa - deficit


def save(split, prob, sem_pred, bundle, threshold, out_dir):
    pred = apply_threshold(prob, sem_pred, threshold)
    indices = bundle.splits[split]
    pred_map = reconstruct_prediction_map(pred, indices, *bundle.labels.shape)
    mask = np.zeros(bundle.labels.shape, dtype=bool)
    mask.reshape(-1)[indices] = True
    metrics = compute_metrics(pred_map, bundle.labels, mask=mask)
    paths = save_prediction_outputs(pred_map, str(out_dir), f"hermiston_usa_{split}", labels=bundle.labels, num_classes=bundle.num_change_classes + 1)
    paths["result_txt"] = write_result_txt(str(out_dir / f"hermiston_usa_{split}_result.txt"), metrics, f"hermiston_usa {split} adaptive-threshold")
    print(split, metrics)
    print(paths)


def main():
    args = parse_args()
    device = torch.device(args.device)
    bundle, loaders = build_hsi_cd_dataloaders(
        dataset="hermiston_usa",
        data_root="dataset/hermiston_USA",
        patch_size=9,
        batch_size=args.batch_size,
        seed=1331,
        train_ratio=0.2,
        val_ratio=0.01,
    )
    model = load_model(args.checkpoint, bundle, device)
    val_prob, val_sem, val_lab = collect(model, loaders["val"], device)
    best = None
    for t in np.linspace(0.10, 0.90, 161):
        pred = apply_threshold(val_prob, val_sem, float(t))
        m = vector_metrics(pred, val_lab)
        s = score(m)
        if best is None or s > best[0]:
            best = (s, float(t), m)
    print("best", best)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "adaptive_threshold.txt").write_text(f"threshold={best[1]:.6f}\nval_metrics={best[2]}\n", encoding="utf-8")
    for split in ("test", "all"):
        prob, sem, _ = collect(model, loaders[split], device)
        save(split, prob, sem, bundle, best[1], out_dir)


if __name__ == "__main__":
    main()
