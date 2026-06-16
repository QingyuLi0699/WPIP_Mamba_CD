"""Tune semantic-rescue two-stage inference for WPIP-Mamba-CD.

Decision rule:
    if P_binary(change) >= t_binary OR max P_semantic(change class) >= t_semantic:
        pred = argmax semantic change class in 1..K
    else:
        pred = 0

This keeps the binary head responsible for no-change while preventing it from
discarding high-confidence semantic changes.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from datasets.hsi_cd_dataset import build_hsi_cd_dataloaders
from models.wpip_mamba import WPIPMambaCD
from train_skeleton import center_labels
from utils.sliding_inference import compute_metrics, reconstruct_prediction_map, save_prediction_outputs, write_result_txt


def parse_args():
    parser = argparse.ArgumentParser(description="Tune semantic-rescue two-stage inference.")
    parser.add_argument("--checkpoint", default="outputs_hermiston_tr20_twostage_c0_45_c1_25/hermiston_usa_best.pt")
    parser.add_argument("--dataset", default="hermiston_usa")
    parser.add_argument("--data-root", default="dataset/hermiston_USA")
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1331)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs_hermiston_rescue_gate")
    return parser.parse_args()


def load_model(checkpoint_path, bundle, device):
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
def collect_split(model, loader, device):
    binary_prob = []
    semantic_pred = []
    semantic_conf = []
    labels = []
    for x1, x2, y in loader:
        x1 = x1.to(device=device, dtype=torch.float32)
        x2 = x2.to(device=device, dtype=torch.float32)
        outputs = model(x1, x2)
        h, w = outputs["final_logits"].shape[-2:]
        final_center = outputs["final_logits"][:, :, h // 2, w // 2]
        binary_center = outputs["binary_logits"][:, :, h // 2, w // 2]
        p_change = torch.softmax(binary_center, dim=1)[:, 1]
        sem_prob = torch.softmax(final_center[:, 1:], dim=1)
        conf, pred = sem_prob.max(dim=1)
        binary_prob.append(p_change.cpu())
        semantic_conf.append(conf.cpu())
        semantic_pred.append((pred + 1).cpu())
        labels.append(center_labels(y).long())
    return {
        "binary_prob": torch.cat(binary_prob).numpy(),
        "semantic_conf": torch.cat(semantic_conf).numpy(),
        "semantic_pred": torch.cat(semantic_pred).numpy().astype(np.int64),
        "labels": torch.cat(labels).numpy().astype(np.int64),
    }


def predict(data, t_binary, t_semantic):
    is_change = (data["binary_prob"] >= t_binary) | (data["semantic_conf"] >= t_semantic)
    return np.where(is_change, data["semantic_pred"], 0).astype(np.int64)


def vector_metrics(pred, labels):
    per = {}
    for cls in sorted(np.unique(labels[labels >= 0]).tolist()):
        mask = labels == cls
        per[int(cls)] = float((pred[mask] == labels[mask]).mean()) if mask.any() else 0.0
    return float((pred == labels).mean()), float(np.mean(list(per.values()))), per


def score(metrics):
    oa, aa, per = metrics
    reference = {
        0: 0.9700,
        1: 0.8800,
        2: 0.9335,
        3: 0.9601,
        4: 0.8821,
        5: 0.9276,
        6: 0.8830,
    }
    deficit = 0.0
    for cls, target in reference.items():
        # C0/C1 are priority targets; other classes are "recover toward 03".
        weight = 4.0 if cls in (0, 1) else 2.0
        deficit += weight * max(0.0, target - per.get(cls, 0.0))
    return aa + 0.25 * oa - deficit


def save_split(split, data, bundle, out_dir, t_binary, t_semantic):
    pred = predict(data, t_binary, t_semantic)
    indices = bundle.splits[split]
    pred_map = reconstruct_prediction_map(pred, indices, *bundle.labels.shape)
    mask = np.zeros(bundle.labels.shape, dtype=bool)
    mask.reshape(-1)[indices] = True
    metrics = compute_metrics(pred_map, bundle.labels, mask=mask)
    paths = save_prediction_outputs(
        pred_map,
        str(out_dir),
        f"hermiston_usa_{split}",
        labels=bundle.labels,
        num_classes=bundle.num_change_classes + 1,
    )
    paths["result_txt"] = write_result_txt(
        str(out_dir / f"hermiston_usa_{split}_result.txt"),
        metrics,
        title=f"hermiston_usa {split} semantic-rescue two-stage",
    )
    print(split, metrics)
    print(paths)


def main():
    args = parse_args()
    device = torch.device(args.device)
    bundle, loaders = build_hsi_cd_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        balanced_train=False,
    )
    model = load_model(args.checkpoint, bundle, device)
    val = collect_split(model, loaders["val"], device)

    candidates = []
    for t_binary in np.linspace(0.15, 0.85, 71):
        for t_semantic in np.linspace(0.45, 0.99, 55):
            pred = predict(val, float(t_binary), float(t_semantic))
            m = vector_metrics(pred, val["labels"])
            candidates.append((score(m), float(t_binary), float(t_semantic), m))
    candidates.sort(key=lambda x: x[0], reverse=True)
    print("top validation candidates")
    for item in candidates[:10]:
        print(f"score={item[0]:.4f} t_binary={item[1]:.3f} t_semantic={item[2]:.3f} metrics={item[3]}")

    _, t_binary, t_semantic, _ = candidates[0]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "gate_config.txt").write_text(
        f"checkpoint={args.checkpoint}\nt_binary={t_binary:.6f}\nt_semantic={t_semantic:.6f}\n",
        encoding="utf-8",
    )
    for split in ("test", "all"):
        data = collect_split(model, loaders[split], device)
        save_split(split, data, bundle, out_dir, t_binary, t_semantic)


if __name__ == "__main__":
    main()
