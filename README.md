# WPIP-Mamba-CD

Wavelet Prior Injected Prototype-guided Mamba Network for end-to-end coarse-to-fine semi-supervised hyperspectral semantic change detection.

## Frozen V1 design

- Feature-level Haar DWT prior
- MambaHSI-style SpeMamba + SpaMamba + BothMamba backbone
- Attention gate prior injection
- Difference + correlation temporal fusion
- Coarse binary head: no-change vs change
- Semantic embedding head: prototype space
- K-prototype memory bank: one prototype per semantic change class
- Prototype-guided pseudo-label assignment
- Entropy-based confidence partition
- Residual Mamba refinement for uncertain regions
- Final K+1 semantic change output

## Input / label convention

Input tensors:

```python
x1, x2: [B, C, H, W]
```

Label tensor:

```python
label: [B, H, W]
-1 = ignored / unlabeled
 0 = no-change
 1..K = semantic change classes
```

## Quick forward test

```bash
conda run -n MambaHSI_env python test_forward.py
```

## Patch-center training

The runnable training skeleton supports local CSANet-style binary change
detection presets (`river`, `china`, `santa`). Labels are mapped into the common
WPIP-Mamba-CD convention:

```python
-1 = ignore / unlabeled
 0 = no-change
 1 = change
```

Each sample is a paired HSI patch. Only the patch center is supervised; all
other positions in the patch label mask are set to `-1`.

```bash
conda run -n MambaHSI_env python train_skeleton.py \
  --dataset river \
  --epochs 1 \
  --batch-size 2 \
  --patch-size 9 \
  --device cpu
```

For real experiments, install the original dependency:

```bash
pip install mamba-ssm==1.2.0
```

If `mamba_ssm` is not installed, or if it is installed but receives CPU tensors
from a CUDA-only build, the code uses a small fallback MLP block only to allow
import/debug/smoke tests. Real experiments should run `mamba_ssm` on CUDA.
