"""Fuse 20% balanced and 20% no-balanced WPIP predictions.

The no-balanced model is used as a no-change expert; the balanced model is used
as the semantic change expert. Class 1 is explicitly preserved from the balanced
model because it is the weakest class under the no-change expert.
"""
from pathlib import Path

import numpy as np

from datasets.hsi_cd_dataset import load_hsi_cd_preset
from utils.sliding_inference import compute_metrics, save_prediction_outputs, write_result_txt


def run(split: str, bundle, out_dir: Path):
    balanced = np.load(f"outputs_hermiston_aug_p025_tr20/hermiston_usa_{split}_pred.npy")
    nobal = np.load(f"outputs_hermiston_aug_p025_tr20_nobal/hermiston_usa_{split}_pred.npy")
    pred = balanced.copy()

    # Use the no-balanced expert only for no-change corrections, but never
    # overwrite class 1 predicted by the balanced semantic-change expert.
    zero_mask = (nobal == 0) & (balanced != 1)
    pred[zero_mask] = 0

    mask = np.zeros(bundle.labels.shape, dtype=bool)
    mask.reshape(-1)[bundle.splits[split]] = True
    metrics = compute_metrics(pred, bundle.labels, mask=mask)
    paths = save_prediction_outputs(
        pred,
        str(out_dir),
        f"hermiston_usa_{split}",
        labels=bundle.labels,
        num_classes=bundle.num_change_classes + 1,
    )
    paths["result_txt"] = write_result_txt(
        str(out_dir / f"hermiston_usa_{split}_result.txt"),
        metrics,
        title=f"hermiston_usa {split} balanced-nobal fusion",
    )
    print(split, metrics)
    print(paths)


def main():
    bundle = load_hsi_cd_preset(
        dataset="hermiston_usa",
        data_root="dataset/hermiston_USA",
        train_ratio=0.2,
        val_ratio=0.01,
        seed=1331,
    )
    out_dir = Path("outputs_hermiston_aug_p025_tr20_balanced_nobal_fusion")
    out_dir.mkdir(exist_ok=True)
    for split in ("test", "all"):
        run(split, bundle, out_dir)


if __name__ == "__main__":
    main()
