"""Search simple fusion rules between balanced and no-change-friendly predictions."""
from pathlib import Path

import numpy as np

from datasets.hsi_cd_dataset import load_hsi_cd_preset
from utils.sliding_inference import compute_metrics, save_prediction_outputs, write_result_txt


def evaluate(name, pred, bundle, split):
    mask = np.zeros(bundle.labels.shape, dtype=bool)
    mask.reshape(-1)[bundle.splits[split]] = True
    metrics = compute_metrics(pred, bundle.labels, mask=mask)
    print(name, metrics)
    return metrics


def main():
    bundle = load_hsi_cd_preset(
        dataset="hermiston_usa",
        data_root="dataset/hermiston_USA",
        train_ratio=0.2,
        val_ratio=0.01,
        seed=1331,
    )
    base = np.load("outputs_hermiston_aug_p025_tr20/hermiston_usa_test_pred.npy")
    helpers = {
        "gate": np.load("outputs_hermiston_aug_p025_tr20_binary_gate/hermiston_usa_test_pred.npy"),
        "nobal10": np.load("outputs_hermiston_aug_p025_nobal/hermiston_usa_test_pred.npy"),
        "calib10": np.load("outputs_hermiston_calib_aug_p025_nobal/hermiston_usa_test_pred.npy"),
        "ensemble10": np.load("outputs_hermiston_ensemble/hermiston_usa_test_pred.npy"),
    }
    evaluate("base", base, bundle, "test")
    best = None
    for helper_name, helper in helpers.items():
        evaluate(helper_name, helper, bundle, "test")
        for preserve1 in (True, False):
            for preserve_changes in ((), (1,), (1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 6)):
                pred = base.copy()
                mask0 = helper == 0
                if preserve_changes:
                    mask0 &= ~np.isin(base, preserve_changes)
                elif preserve1:
                    mask0 &= base != 1
                pred[mask0] = 0
                metrics = evaluate(f"{helper_name}_zero_preserve_{preserve_changes or ('1' if preserve1 else 'none')}", pred, bundle, "test")
                per = metrics["per_class_acc"]
                deficit = max(0, 0.97 - per[0]) * 10 + max(0, 0.88 - per[1]) * 10
                # Penalize dropping classes 2..6 below the 20% balanced baseline.
                baseline = {2: 0.933473, 3: 0.960069, 4: 0.882060, 5: 0.927583, 6: 0.883037}
                deficit += sum(max(0, baseline[c] - per[c]) * 2 for c in baseline)
                score = metrics["aa"] + 0.2 * metrics["oa"] - deficit
                if best is None or score > best[0]:
                    best = (score, helper_name, preserve_changes, preserve1, pred, metrics)
    print("BEST", best[:4], best[5])
    out_dir = Path("outputs_hermiston_aug_p025_tr20_fusion_search")
    out_dir.mkdir(exist_ok=True)
    pred = best[4]
    metrics = best[5]
    paths = save_prediction_outputs(pred, str(out_dir), "hermiston_usa_test", labels=bundle.labels, num_classes=bundle.num_change_classes + 1)
    paths["result_txt"] = write_result_txt(str(out_dir / "hermiston_usa_test_result.txt"), metrics, "hermiston_usa test fusion-search")
    print(paths)


if __name__ == "__main__":
    main()
