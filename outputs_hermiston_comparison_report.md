# Hermiston_USA WPIP-Mamba-CD Comparison Report

Training protocol used for the main comparison:

- Dataset: `dataset/hermiston_USA`
- Classes: `0 = no-change`, `1..6 = change types`
- Train/val split: `train_ratio=0.1`, `val_ratio=0.01`
- Test samples: remaining labeled pixels
- Metrics reported: per-class accuracy, OA, AA, KC
- IoU is intentionally omitted.

## MambaHSI Reference

Reference file:
`Comparison/MambaHSI-main/RUNS_concat_compare_noiou_e150_7cls/MambaHSI/Hermiston/mean_result.txt`

MambaHSI 5-run mean:

| Method | OA | AA | KC |
|---|---:|---:|---:|
| MambaHSI | 0.7942 | 0.8413 | 0.5784 |

Mean per-class Acc:

| Class | Acc |
|---:|---:|
| 0 | 0.7763 |
| 1 | 0.8086 |
| 2 | 0.8532 |
| 3 | 0.8918 |
| 4 | 0.8620 |
| 5 | 0.9227 |
| 6 | 0.7748 |

## WPIP-Mamba-CD Results

### Balanced + Augmented Variant

Output directory: `outputs_hermiston_aug_p025`

This variant uses class-balanced sampling, moderate class weighting
(`class_weight_power=0.25`), and random flip/rotation patch augmentation.

| Method | OA | AA | KC |
|---|---:|---:|---:|
| WPIP-Mamba-CD balanced+aug | 0.8899 | 0.8931 | 0.7542 |

Per-class Acc:

| Class | Acc |
|---:|---:|
| 0 | 0.8893 |
| 1 | 0.8345 |
| 2 | 0.9043 |
| 3 | 0.8847 |
| 4 | 0.8784 |
| 5 | 0.9660 |
| 6 | 0.8944 |

### Balanced + Augmented Variant With 20% Training Samples

Output directory: `outputs_hermiston_aug_p025_tr20`

This run keeps the same single-model balanced+aug setting, but increases the
training samples to 20% of each category while keeping `val_ratio=0.01`.

| Method | OA | AA | KC |
|---|---:|---:|---:|
| WPIP-Mamba-CD balanced+aug, 20% train | 0.8941 | 0.9061 | 0.7650 |

Per-class Acc:

| Class | Acc |
|---:|---:|
| 0 | 0.8864 |
| 1 | 0.8702 |
| 2 | 0.9335 |
| 3 | 0.9601 |
| 4 | 0.8821 |
| 5 | 0.9276 |
| 6 | 0.8830 |

Compared with the 10% balanced+aug run, the 20% run improves OA, AA, KC, and
most semantic change classes. The main tradeoff is class 5, which remains strong
but decreases from `0.9660` to `0.9276`.

### Highest OA/KC Ensemble

Output directory: `outputs_hermiston_ensemble`

This majority-vote ensemble combines saved predictions from the trained WPIP
variants. It gives the strongest OA/KC and full-map visualization quality.

| Method | OA | AA | KC |
|---|---:|---:|---:|
| WPIP-Mamba-CD ensemble | 0.9352 | 0.8136 | 0.8338 |

Per-class Acc:

| Class | Acc |
|---:|---:|
| 0 | 0.9657 |
| 1 | 0.8001 |
| 2 | 0.7682 |
| 3 | 0.9255 |
| 4 | 0.6853 |
| 5 | 0.9112 |
| 6 | 0.6393 |

## Saved Visual Outputs

Balanced+augmented model:

- `outputs_hermiston_aug_p025/hermiston_usa_test_pred.png`
- `outputs_hermiston_aug_p025/hermiston_usa_all_pred.png`
- `outputs_hermiston_aug_p025/hermiston_usa_test_result.txt`
- `outputs_hermiston_aug_p025/hermiston_usa_all_result.txt`

Balanced+augmented model with 20% training samples:

- `outputs_hermiston_aug_p025_tr20/hermiston_usa_test_pred.png`
- `outputs_hermiston_aug_p025_tr20/hermiston_usa_all_pred.png`
- `outputs_hermiston_aug_p025_tr20/hermiston_usa_test_result.txt`
- `outputs_hermiston_aug_p025_tr20/hermiston_usa_all_result.txt`

Ensemble:

- `outputs_hermiston_ensemble/hermiston_usa_test_pred.png`
- `outputs_hermiston_ensemble/hermiston_usa_all_pred.png`
- `outputs_hermiston_ensemble/hermiston_usa_test_result.txt`
- `outputs_hermiston_ensemble/hermiston_usa_all_result.txt`

## Current Recommendation

For a TGRS-style main table under 10% training samples, use
`outputs_hermiston_aug_p025` as the primary single-model result because it
improves OA, AA, and KC over the MambaHSI reference and is more class-balanced.

For the 20% training-sample setting, use `outputs_hermiston_aug_p025_tr20`; it
is the strongest balanced single-model result so far.

For a best visual/full-map result, use `outputs_hermiston_ensemble` because it
achieves the highest OA and KC.
