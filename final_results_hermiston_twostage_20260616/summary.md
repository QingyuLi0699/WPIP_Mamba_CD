# Hermiston_USA Final Strict Two-Stage Results

## Training Principle

The final experiments follow the strict end-to-end two-stage setting:

- `binary_head` learns `0 = no-change` vs `change`.
- The semantic classification branch is supervised only on `label > 0`.
- No-change pixels do not participate in the semantic `1..K` change-category loss.
- Training is still end-to-end: one forward pass, one total loss, one backward pass, one optimizer step.

Common protocol:

- Dataset: `dataset/hermiston_USA`
- Labels: `0 = no-change`, `1..6 = semantic change classes`
- Train samples: 20% of each category
- Validation samples: 1% of each category
- Metrics: per-class accuracy, OA, AA, KC
- IoU is not reported.

## Result Summary

| Variant | OA | AA | KC | C0 | C1 | Notes |
|---|---:|---:|---:|---:|---:|---|
| Strict two-stage C1 emphasis | 0.9249 | 0.8691 | 0.8182 | 0.9398 | 0.9054 | Best strict single-model class-1 result |
| C0/C1 fusion | 0.9370 | 0.8310 | 0.8387 | 0.9665 | 0.9054 | Best C0/C1 tradeoff, uses prediction fusion |
| Balanced+aug 20% reference | 0.8941 | 0.9061 | 0.7650 | 0.8864 | 0.8702 | Best AA / balanced semantic classes |
| C0 expert reference | 0.9436 | 0.8436 | 0.8554 | 0.9706 | 0.7861 | Best class-0 single-model reference |
| Semantic-rescue other-classes | 0.8827 | 0.8978 | 0.7425 | 0.8753 | 0.9464 | Recovers semantic change classes via rescue gate |

## Recommended Use

For the method description and main single-model table, use:

`01_strict_twostage_c1_emphasis`

This is the cleanest version of the proposed idea: no-change is excluded from
semantic change classification, while the whole network is still trained
end-to-end.

For the best class-0/class-1 practical tradeoff and visualization, use:

`02_c0_c1_fusion`

It reaches class 0 close to the requested 97% target and class 1 above the
requested 88% target, but it is a prediction-level fusion of two trained
two-stage variants.

For diagnosing whether the semantic branch can recover the nonzero change
classes, use:

`05_semantic_rescue_other_classes`

This variant lets high-confidence semantic changes override the binary gate.
It restores class 1 and most other change categories close to the balanced
reference, but sacrifices class 0. This confirms that the lower change-class
accuracy in strict two-stage inference is mainly caused by binary false
negatives rather than a completely weak semantic branch.

## Folder Contents

Each subfolder contains:

- `hermiston_usa_test_result.txt`
- `hermiston_usa_all_result.txt`
- `hermiston_usa_test_pred.png`
- `hermiston_usa_all_pred.png`
- `hermiston_usa_test_gt.png`

Subfolders:

- `01_strict_twostage_c1_emphasis`: strict end-to-end two-stage, C1-emphasized sampler.
- `02_c0_c1_fusion`: fuses the C1-emphasized semantic expert with the C0 expert.
- `03_balanced_aug_20_reference`: previous balanced+aug 20% reference.
- `04_c0_expert_reference`: no-balanced 20% reference, strongest class-0 model.
- `05_semantic_rescue_other_classes`: semantic-rescue two-stage inference, strongest change-class recovery.
