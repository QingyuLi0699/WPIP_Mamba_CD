"""Ensemble saved Hermiston_USA prediction maps by majority vote."""
from pathlib import Path

import numpy as np

from datasets.hsi_cd_dataset import load_hsi_cd_preset
from utils.sliding_inference import compute_metrics, save_prediction_outputs, write_result_txt


def majority_vote(stack: np.ndarray) -> np.ndarray:
    """Majority vote over [N,H,W] integer prediction maps."""
    n, h, w = stack.shape
    out = np.full((h, w), -1, dtype=np.int64)
    for i in range(h):
        for j in range(w):
            vals = stack[:, i, j]
            vals = vals[vals >= 0]
            if vals.size == 0:
                continue
            counts = np.bincount(vals)
            out[i, j] = int(np.argmax(counts))
    return out


def main():
    bundle = load_hsi_cd_preset(
        dataset="hermiston_usa",
        data_root="dataset/hermiston_USA",
        train_ratio=0.1,
        val_ratio=0.01,
        seed=1331,
    )
    dirs = [
        "outputs_hermiston_improved",
        "outputs_hermiston_aug_p025",
        "outputs_hermiston_aug_p025_nobal",
        "outputs_hermiston_calib_aug_p025_nobal",
    ]
    out_dir = "outputs_hermiston_ensemble"
    num_classes = bundle.num_change_classes + 1

    for split in ("test", "all"):
        arrays = [np.load(Path(d) / f"hermiston_usa_{split}_pred.npy") for d in dirs]
        pred = majority_vote(np.stack(arrays, axis=0))
        if split == "test":
            mask = np.zeros(bundle.labels.shape, dtype=bool)
            mask.reshape(-1)[bundle.splits["test"]] = True
        else:
            mask = bundle.labels >= 0
        metrics = compute_metrics(pred, bundle.labels, mask=mask)
        paths = save_prediction_outputs(
            pred,
            out_dir,
            f"hermiston_usa_{split}",
            labels=bundle.labels,
            num_classes=num_classes,
        )
        paths["result_txt"] = write_result_txt(
            str(Path(out_dir) / f"hermiston_usa_{split}_result.txt"),
            metrics,
            title=f"hermiston_usa {split} ensemble",
        )
        print(split, metrics)
        print(paths)


if __name__ == "__main__":
    main()
