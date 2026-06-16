"""Fuse strict two-stage class-1 and class-0 expert predictions."""
from pathlib import Path

import numpy as np

from datasets.hsi_cd_dataset import load_hsi_cd_preset
from utils.sliding_inference import compute_metrics, save_prediction_outputs, write_result_txt


def main():
    bundle = load_hsi_cd_preset(
        dataset="hermiston_usa",
        data_root="dataset/hermiston_USA",
        train_ratio=0.2,
        val_ratio=0.01,
        seed=1331,
    )
    out = Path("outputs_hermiston_tr20_twostage_c1_c0_fusion")
    out.mkdir(exist_ok=True)
    for split in ("test", "all"):
        c1 = np.load(f"outputs_hermiston_tr20_twostage_c0_45_c1_25/hermiston_usa_{split}_pred.npy")
        c0 = np.load(f"outputs_hermiston_aug_p025_tr20_nobal/hermiston_usa_{split}_pred.npy")
        pred = c1.copy()
        pred[(c0 == 0) & (c1 != 1)] = 0

        mask = np.zeros(bundle.labels.shape, dtype=bool)
        mask.reshape(-1)[bundle.splits[split]] = True
        metrics = compute_metrics(pred, bundle.labels, mask=mask)
        paths = save_prediction_outputs(
            pred,
            str(out),
            f"hermiston_usa_{split}",
            labels=bundle.labels,
            num_classes=bundle.num_change_classes + 1,
        )
        paths["result_txt"] = write_result_txt(
            str(out / f"hermiston_usa_{split}_result.txt"),
            metrics,
            f"hermiston_usa {split} c1-c0 fusion",
        )
        print(split, metrics)
        print(paths)


if __name__ == "__main__":
    main()
